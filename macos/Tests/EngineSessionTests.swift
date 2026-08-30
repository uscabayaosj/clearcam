import XCTest
@testable import ClearCam

final class EngineSessionTests: XCTestCase {
    func testOnlyOwnedLoopbackOriginIsAllowed() {
        let session = EngineSession(url: URL(string: "http://127.0.0.1:54321/")!, token: "test")
        XCTAssertTrue(session.allows(URL(string: "http://127.0.0.1:54321/event_thumbs")!))
        for bad in ["http://127.0.0.1:8080/", "https://example.com/", "file:///etc/passwd", "http://localhost:54321/"] {
            XCTAssertFalse(session.allows(URL(string: bad)!))
        }
    }
    func testAuthorizationIsNotInURL() {
        let request = EngineSession(url: URL(string: "http://127.0.0.1:54321/")!, token: "test-secret").request("engine_status")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-secret")
        XCTAssertFalse(request.url!.absoluteString.contains("test-secret"))
    }
}
