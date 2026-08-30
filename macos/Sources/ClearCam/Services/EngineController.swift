import Foundation
import AppKit
import Combine
import UserNotifications

@MainActor
final class EngineController: ObservableObject {
    static let shared = EngineController()
    @Published private(set) var session: EngineSession?
    @Published private(set) var status = "Starting local engine"
    @Published private(set) var failure: String?
    @Published private(set) var starting = false
    @Published private(set) var notificationsEnabled = false
    @Published private(set) var notificationsPausedUntil: Date?

    let support: URL
    private var process: Process?
    private var monitor: Task<Void, Never>?
    private var logHandle: FileHandle?
    private var intentionalStop = false
    private var restartAttempts = 0
    private let network = URLSession(configuration: .ephemeral)

    init() {
        support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ClearCam", isDirectory: true)
    }

    func start() async {
        guard process == nil, !starting else { return }
        starting = true
        intentionalStop = false
        failure = nil
        status = "Starting local engine"
        defer { starting = false }
        do {
            let fm = FileManager.default
            try fm.createDirectory(at: support, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let runtime = support.appendingPathComponent("Runtime", isDirectory: true)
            try fm.createDirectory(at: runtime, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let readyFile = runtime.appendingPathComponent("ready-\(UUID().uuidString).json")
            defer { try? fm.removeItem(at: readyFile) }
            guard let resources = Bundle.main.resourceURL else { throw HubError("App resources are missing.") }
            let python = resources.appendingPathComponent("Runtime/bin/python3.11")
            let bootstrap = resources.appendingPathComponent("Engine/macos/engine_bootstrap.py")
            guard fm.isExecutableFile(atPath: python.path), fm.fileExists(atPath: bootstrap.path) else {
                throw HubError("The bundled engine is missing. Rebuild or reinstall this alpha.")
            }
            let child = Process()
            child.executableURL = python
            child.arguments = ["-u", bootstrap.path]
            child.currentDirectoryURL = resources.appendingPathComponent("Engine")
            let token = UUID().uuidString + UUID().uuidString
            var env = ProcessInfo.processInfo.environment
            // Never inherit developer Python paths or depend on Homebrew.
            for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "DYLD_LIBRARY_PATH"] { env.removeValue(forKey: key) }
            env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["CLEARCAM_NATIVE"] = "1"
            env["CLEARCAM_PARENT_PID"] = String(ProcessInfo.processInfo.processIdentifier)
            env["CLEARCAM_DATA_DIR"] = support.appendingPathComponent("Data").path
            env["CLEARCAM_READY_FILE"] = readyFile.path
            env["CLEARCAM_SESSION_TOKEN"] = token
            env["CLEARCAM_BIND_HOST"] = "127.0.0.1"
            env["CLEARCAM_PORT"] = "0"
            env["CLEARCAM_MODEL_DIR"] = resources.appendingPathComponent("Models").path
            env["CLEARCAM_FFMPEG"] = resources.appendingPathComponent("Tools/ffmpeg").path
            env["CACHEDB"] = support.appendingPathComponent("Caches/tinygrad.db").path
            env["XDG_CACHE_HOME"] = support.appendingPathComponent("Caches").path
            child.environment = env
            let logURL = support.appendingPathComponent("engine.log")
            if !fm.fileExists(atPath: logURL.path) { fm.createFile(atPath: logURL.path, contents: nil, attributes: [.posixPermissions: 0o600]) }
            let log = try FileHandle(forWritingTo: logURL)
            try log.seekToEnd()
            logHandle = log
            child.standardOutput = log
            child.standardError = log
            child.terminationHandler = { [weak self] stopped in
                Task { @MainActor in await self?.didExit(stopped) }
            }
            process = child
            try child.run()
            for _ in 0..<240 {
                guard child.isRunning, !intentionalStop else { throw HubError("The engine stopped during startup. Open the engine log for details.") }
                if let data = try? Data(contentsOf: readyFile), let ready = try? JSONDecoder().decode(EngineReady.self, from: data), ready.pid == child.processIdentifier, (1...65535).contains(ready.port) {
                    let candidate = EngineSession(url: URL(string: "http://127.0.0.1:\(ready.port)/")!, token: token)
                    if let (_, response) = try? await network.data(for: candidate.request("engine_status")), (response as? HTTPURLResponse)?.statusCode == 200 {
                        session = candidate
                        status = "Local engine running"
                        startMonitoring(candidate)
                        return
                    }
                }
                try await Task.sleep(for: .milliseconds(500))
            }
            throw HubError("The engine did not become ready within two minutes. Open the engine log, then retry.")
        } catch {
            failure = error.localizedDescription
            status = "Engine needs attention"
            await stop(keepFailure: true)
        }
    }

    func stop(keepFailure: Bool = false) async {
        intentionalStop = true
        monitor?.cancel()
        monitor = nil
        session = nil
        guard let child = process else { return }
        if child.isRunning { child.terminate() }
        for _ in 0..<70 {
            if !child.isRunning { break }
            try? await Task.sleep(for: .milliseconds(500))
        }
        if child.isRunning {
            // The bootstrap owns a separate process group; never target other apps.
            kill(-child.processIdentifier, SIGKILL)
        }
        process = nil
        try? logHandle?.close()
        logHandle = nil
        if !keepFailure { status = "Engine stopped" }
    }

    private func didExit(_ child: Process) async {
        guard process === child else { return }
        process = nil
        session = nil
        monitor?.cancel()
        guard !intentionalStop, !starting else { return }
        restartAttempts += 1
        guard restartAttempts <= 3 else {
            status = "Engine needs attention"
            failure = "The engine stopped repeatedly. Open the log before retrying."
            return
        }
        status = "Reconnecting local engine"
        try? await Task.sleep(for: .seconds(2))
        guard !intentionalStop else { return }
        await start()
    }

    private func startMonitoring(_ active: EngineSession) {
        monitor?.cancel()
        monitor = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                do {
                    let (data, response) = try await self.network.data(for: active.request("engine_status"))
                    guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw HubError("Engine connection unavailable") }
                    let health = try JSONDecoder().decode(HubStatus.self, from: data)
                    self.status = health.cameras.isEmpty ? "Ready to add a camera" : health.cameras.values.allSatisfy { $0.state == "detecting" } ? "Recording · detection running" : "Camera needs attention"
                    let (notices, _) = try await self.network.data(for: active.request("native_notifications"))
                    let batch = try JSONDecoder().decode(NotificationBatch.self, from: notices)
                    for event in batch.notifications { await self.deliver(event) }
                } catch { if !Task.isCancelled { self.status = "Checking engine connection" } }
                try? await Task.sleep(for: .seconds(3))
            }
        }
    }

    func retry() async { restartAttempts = 0; await stop(); await start() }

    /// After system wake, camera connections are dead; verify the engine is
    /// reachable and restart it if not, rather than waiting out slow recovery.
    func recoverAfterWake() async {
        restartAttempts = 0
        guard let active = session else { await start(); return }
        try? await Task.sleep(for: .seconds(5))  // let the network come back first
        if let (_, response) = try? await network.data(for: active.request("engine_status")),
           (response as? HTTPURLResponse)?.statusCode == 200 { return }
        status = "Restarting after sleep"
        await stop()
        await start()
    }

    func pauseNotifications(minutes: Int) async {
        guard let active = session else { return }
        _ = try? await network.data(for: active.request("pause_notifications?minutes=\(minutes)"))
        notificationsPausedUntil = minutes > 0 ? Date().addingTimeInterval(TimeInterval(minutes) * 60) : nil
    }
    func revealData() { NSWorkspace.shared.open(support.appendingPathComponent("Data")) }
    func revealLog() { NSWorkspace.shared.open(support.appendingPathComponent("engine.log")) }
    func enableNotifications() async {
        do { notificationsEnabled = try await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) }
        catch { failure = "Notifications could not be enabled: \(error.localizedDescription)" }
    }
    private func deliver(_ event: NativeNotification) async {
        let content = UNMutableNotificationContent()
        content.title = event.title
        content.body = event.body
        content.sound = .default
        try? await UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: event.id, content: content, trigger: nil))
    }
}

struct HubError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
