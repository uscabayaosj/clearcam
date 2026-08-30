import SwiftUI
import WebKit

struct CameraWebView: NSViewRepresentable {
    let session: EngineSession
    func makeCoordinator() -> Coordinator { Coordinator(session: session) }
    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        config.mediaTypesRequiringUserActionForPlayback = []
        let web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = context.coordinator
        web.uiDelegate = context.coordinator
        web.allowsBackForwardNavigationGestures = false
        let cookie = HTTPCookie(properties: [
            .domain: "127.0.0.1", .path: "/", .name: "ClearCamSession", .value: session.token,
            HTTPCookiePropertyKey("HttpOnly"): "TRUE", HTTPCookiePropertyKey("SameSite"): "Strict"
        ])!
        config.websiteDataStore.httpCookieStore.setCookie(cookie) { web.load(URLRequest(url: session.url)) }
        return web
    }
    func updateNSView(_ web: WKWebView, context: Context) {}
    static func dismantleNSView(_ web: WKWebView, coordinator: Coordinator) { web.stopLoading(); web.navigationDelegate = nil; web.uiDelegate = nil }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        let session: EngineSession
        init(session: EngineSession) { self.session = session }
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let url = action.request.url else { decisionHandler(.cancel); return }
            if session.allows(url) { decisionHandler(.allow) }
            else {
                if action.navigationType == .linkActivated, url.scheme == "https" { NSWorkspace.shared.open(url) }
                decisionHandler(.cancel)
            }
        }
        func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
            let alert = NSAlert(); alert.messageText = "ClearCam"; alert.informativeText = message
            alert.addButton(withTitle: "OK")
            if let window = webView.window { alert.beginSheetModal(for: window) { _ in completionHandler() } }
            else { completionHandler() }
        }
        func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
            let alert = NSAlert(); alert.messageText = "Confirm change"; alert.informativeText = message
            alert.addButton(withTitle: "Continue"); alert.addButton(withTitle: "Cancel")
            if let window = webView.window { alert.beginSheetModal(for: window) { completionHandler($0 == .alertFirstButtonReturn) } }
            else { completionHandler(false) }
        }
    }
}
