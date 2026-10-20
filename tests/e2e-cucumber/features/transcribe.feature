Feature: Transcribing a recording

  # `rocm transcribe` sends a file to a speech-to-text server `rocm serve`
  # started. These run on every lane against a mock speech-to-text server; the
  # real Whisper path is serve-27's subject.

  # The help text promises the recording is only read: nothing is written to
  # disk and no audio is kept. The recording carries a marker no other file
  # could contain, so finding it anywhere under the CLI's data, config and cache
  # roots afterwards would prove the promise false.
  @id:transcribe-prints-what-was-said
  Scenario: transcribe-01 - A recording is transcribed by the running speech-to-text server
    Given a speech-to-text model is being served and registered with the CLI
    When the user transcribes a recording
    Then the transcript of the recording is printed
    And the speech-to-text server received the recording
    And no copy of the recording was written to disk

  @id:transcribe-without-a-speech-server
  Scenario: transcribe-02 - Transcribing with only a chat server running names the command that starts one
    Given a model is being served
    And the model is registered with the CLI
    When the user transcribes a recording
    Then transcribing is refused because no speech-to-text server is running
