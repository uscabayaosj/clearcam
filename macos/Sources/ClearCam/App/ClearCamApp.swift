import SwiftUI
import AppKit

@main
struct ClearCamApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var engine = EngineController.shared

    var body: some Scene {
        WindowGroup("ClearCam", id: "main") {
            ContentView(engine: engine)
                .frame(minWidth: 900, minHeight: 640)
                .task { await engine.start() }
        }
        .defaultSize(width: 1180, height: 820)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Camera Hub") {
                Button("Show Recordings in Finder") { engine.revealData() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                Button("Show Engine Log") { engine.revealLog() }
            }
        }
        Settings { SettingsView(engine: engine) }
        MenuBarExtra("ClearCam", systemImage: "video.circle") {
            HubMenu(engine: engine)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        Task {
            await EngineController.shared.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if !hasVisibleWindows { sender.windows.first(where: { $0.canBecomeMain })?.makeKeyAndOrderFront(nil) }
        return true
    }
}
