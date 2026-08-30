// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ClearCam",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "ClearCam", targets: ["ClearCam"])],
    targets: [
        .executableTarget(name: "ClearCam", path: "Sources/ClearCam"),
        .testTarget(name: "ClearCamTests", dependencies: ["ClearCam"], path: "Tests")
    ]
)
