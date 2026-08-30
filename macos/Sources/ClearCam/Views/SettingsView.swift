import SwiftUI
import ServiceManagement

struct SettingsView: View {
    @ObservedObject var engine: EngineController
    @State private var loginEnabled = SMAppService.mainApp.status == .enabled
    @State private var message = ""

    var body: some View {
        Form {
            Section("On this Mac") {
                Toggle("Open ClearCam at login", isOn: $loginEnabled)
                    .onChange(of: loginEnabled) { _, enabled in
                        do {
                            if enabled { try SMAppService.mainApp.register() }
                            else { try SMAppService.mainApp.unregister() }
                            message = SMAppService.mainApp.status == .requiresApproval ? "Allow ClearCam in System Settings → Login Items." : ""
                        } catch {
                            message = error.localizedDescription
                            loginEnabled = SMAppService.mainApp.status == .enabled
                        }
                    }
                Text("The engine runs while ClearCam is open. Closing its window keeps recording; quitting stops it.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Notifications") {
                Button("Enable Camera Notifications…") { Task { await engine.enableNotifications() } }
                Text("Allow alerts from ClearCam when macOS asks. Notification delivery also depends on Focus settings.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Private storage") {
                Button("Show Data Folder") { engine.revealData() }
                Text(engine.support.appendingPathComponent("Data").path).font(.caption).textSelection(.enabled)
                Text("Qwen 2B and YOLO tiny are included. Search models and other model sizes are not included in this alpha.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !message.isEmpty { Text(message).foregroundStyle(.secondary) }
        }
        .formStyle(.grouped).frame(width: 520, height: 440)
    }
}
