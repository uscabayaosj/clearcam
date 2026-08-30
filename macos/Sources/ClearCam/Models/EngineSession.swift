import Foundation

struct EngineSession: Equatable {
    let url: URL
    let token: String

    func allows(_ candidate: URL) -> Bool {
        candidate.scheme == "http" && candidate.host == "127.0.0.1" && candidate.port == url.port
    }

    func request(_ path: String) -> URLRequest {
        var request = URLRequest(url: url.appendingPathComponent(path))
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 5
        return request
    }
}

struct EngineReady: Decodable { let port: Int; let pid: Int32 }
struct HubStatus: Decodable {
    struct Camera: Decodable { let state: String }
    let cameras: [String: Camera]
}
struct NativeNotification: Decodable { let id: String; let title: String; let body: String }
struct NotificationBatch: Decodable { let notifications: [NativeNotification] }
