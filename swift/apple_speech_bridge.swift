import AVFoundation
import Foundation
import FoundationModels
import Speech

private enum BridgeFailure: LocalizedError {
    case requiresMacOS26
    case appleIntelligenceUnavailable(String)
    case speechUnavailable
    case unsupportedLocale
    case assetInstallationFailed
    case audioReadFailed
    case transcriptionFailed

    var code: Int32 {
        switch self {
        case .requiresMacOS26: return 1
        case .appleIntelligenceUnavailable: return 2
        case .speechUnavailable: return 3
        case .unsupportedLocale: return 4
        case .assetInstallationFailed: return 5
        case .audioReadFailed: return 6
        case .transcriptionFailed: return 7
        }
    }

    var errorDescription: String? {
        switch self {
        case .requiresMacOS26:
            return "Apple Speech requires macOS 26 or later"
        case .appleIntelligenceUnavailable(let reason):
            return "Apple Speech requires available Apple Intelligence: \(reason)"
        case .speechUnavailable:
            return "Apple SpeechTranscriber is unavailable on this device"
        case .unsupportedLocale:
            return "Apple Speech does not support the requested locale"
        case .assetInstallationFailed:
            return "Apple Speech could not install the required locale asset"
        case .audioReadFailed:
            return "Apple Speech could not read the input media file"
        case .transcriptionFailed:
            return "Apple Speech transcription failed"
        }
    }
}

@available(macOS 26.0, *)
private struct BridgeSegment: Encodable {
    let start: Double
    let end: Double
    let text: String
}

@available(macOS 26.0, *)
private struct BridgePayload: Encodable {
    let locale: String
    let assetInstallRequested: Bool
    let segments: [BridgeSegment]
}

@available(macOS 26.0, *)
private func requireAppleIntelligence() throws {
    switch SystemLanguageModel.default.availability {
    case .available:
        return
    case .unavailable(.deviceNotEligible):
        throw BridgeFailure.appleIntelligenceUnavailable("device not eligible")
    case .unavailable(.appleIntelligenceNotEnabled):
        throw BridgeFailure.appleIntelligenceUnavailable("not enabled")
    case .unavailable(.modelNotReady):
        throw BridgeFailure.appleIntelligenceUnavailable("model not ready")
    case .unavailable:
        throw BridgeFailure.appleIntelligenceUnavailable("unknown reason")
    }
}

@available(macOS 26.0, *)
private func runTranscription(url: URL, localeIdentifier: String?) async throws -> String {
    try requireAppleIntelligence()
    guard SpeechTranscriber.isAvailable else {
        throw BridgeFailure.speechUnavailable
    }

    let requestedLocale = localeIdentifier.map(Locale.init(identifier:)) ?? Locale.current
    guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: requestedLocale) else {
        throw BridgeFailure.unsupportedLocale
    }
    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)

    var assetInstallRequested = false
    do {
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            assetInstallRequested = true
            try await request.downloadAndInstall()
        }
    } catch {
        throw BridgeFailure.assetInstallationFailed
    }

    let audioFile: AVAudioFile
    do {
        audioFile = try AVAudioFile(forReading: url)
    } catch {
        throw BridgeFailure.audioReadFailed
    }

    let analyzer = SpeechAnalyzer(modules: [transcriber])
    var segments: [BridgeSegment] = []
    let collector = Task {
        for try await result in transcriber.results {
            let start = result.range.start.seconds
            let end = result.range.end.seconds
            segments.append(BridgeSegment(
                start: start.isFinite ? start : 0,
                end: end.isFinite ? end : 0,
                text: String(result.text.characters)
            ))
        }
    }

    do {
        _ = try await analyzer.analyzeSequence(from: audioFile)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        try await collector.value
    } catch {
        collector.cancel()
        await analyzer.cancelAndFinishNow()
        throw BridgeFailure.transcriptionFailed
    }

    let payload = BridgePayload(
        locale: locale.identifier,
        assetInstallRequested: assetInstallRequested,
        segments: segments
    )
    let data = try JSONEncoder().encode(payload)
    return String(decoding: data, as: UTF8.self)
}

@_cdecl("transcribeit_apple_speech_transcribe_file")
public func transcribeitAppleSpeechTranscribeFile(
    path: UnsafePointer<CChar>,
    locale: UnsafePointer<CChar>?,
    userData: UnsafeMutableRawPointer?,
    onDone: @convention(c) (UnsafePointer<UInt8>?, Int, UnsafeMutableRawPointer?) -> Void,
    onError: @convention(c) (Int32, UnsafePointer<UInt8>?, Int, UnsafeMutableRawPointer?) -> Void
) {
    guard #available(macOS 26.0, *) else {
        let failure = BridgeFailure.requiresMacOS26
        let bytes = Array(failure.localizedDescription.utf8)
        bytes.withUnsafeBufferPointer { onError(failure.code, $0.baseAddress, $0.count, userData) }
        return
    }

    let url = URL(fileURLWithPath: String(cString: path))
    let localeIdentifier = locale.map { String(cString: $0) }
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            let result = try await runTranscription(url: url, localeIdentifier: localeIdentifier)
            let bytes = Array(result.utf8)
            bytes.withUnsafeBufferPointer { onDone($0.baseAddress, $0.count, userData) }
        } catch {
            let failure = error as? BridgeFailure
            let bytes = Array(error.localizedDescription.utf8)
            bytes.withUnsafeBufferPointer {
                onError(failure?.code ?? 7, $0.baseAddress, $0.count, userData)
            }
        }
        semaphore.signal()
    }
    semaphore.wait()
}
