import SwiftUI

struct ContentView: View {
    @ObservedObject var engine: EngineController
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: engine.session == nil ? "circle.dotted" : "checkmark.shield")
                    .foregroundStyle(engine.session == nil ? Color.secondary : Color.green)
                Text(engine.status).font(.callout)
                Spacer()
                Text("MAC ALPHA").font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                SettingsLink { Image(systemName: "gearshape") }.help("ClearCam settings")
            }
            .padding(.horizontal, 20).padding(.vertical, 10).background(.bar)
            Divider()
            if let session = engine.session {
                CameraWebView(session: session).id(session.url)
            } else {
                VStack(spacing: 18) {
                    Image(systemName: "video.circle").font(.system(size: 52, weight: .light)).foregroundStyle(.secondary)
                    Text(engine.failure == nil ? "Your home, on this Mac." : "Let’s reconnect.")
                        .font(.title2.weight(.semibold))
                    Text(engine.failure ?? "Starting your private camera engine. Your recordings and AI stay local.")
                        .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 420)
                    if engine.starting { ProgressView().controlSize(.small) }
                    else {
                        HStack {
                            Button("Open Engine Log") { engine.revealLog() }
                            Button("Try Again") { Task { await engine.retry() } }.buttonStyle(.borderedProminent)
                        }
                    }
                }.frame(maxWidth: .infinity, maxHeight: .infinity).padding(40)
            }
        }
    }
}

struct HubMenu: View {
    @ObservedObject var engine: EngineController
    @Environment(\.openWindow) private var openWindow
    var body: some View {
        Text(engine.session == nil ? "Engine unavailable" : "Local engine running")
        Button("Open ClearCam") { openWindow(id: "main"); NSApp.activate(ignoringOtherApps: true) }
        Button("Show Recordings") { engine.revealData() }
        if let until = engine.notificationsPausedUntil, until > Date() {
            Button("Resume notifications (paused until \(until.formatted(date: .omitted, time: .shortened)))") {
                Task { await engine.pauseNotifications(minutes: 0) }
            }
        } else {
            Button("Pause notifications for 1 hour") {
                Task { await engine.pauseNotifications(minutes: 60) }
            }
        }
        SettingsLink { Text("Settings…") }
        Divider()
        Text("Closing the window keeps recording")
            .font(.caption)
        Button("Quit & Stop Recording") { NSApp.terminate(nil) }
    }
}
