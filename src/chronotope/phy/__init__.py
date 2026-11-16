"""PHY — the physical two-machine path (cards/PHY-physical-path.md).

One clock. Everything is stamped on this machine's CLOCK_MONOTONIC (time.monotonic_ns). The target
machine runs one instrumented probe page served from here; the page synchronizes to this clock over
LAN (Cristian's algorithm against /t, min-RTT sample) and encodes agent-clock timestamps into the
pixels it draws, so that a frame read back through the capture card carries the time it was drawn
and the time the last input was received — both on the agent clock.

  E→O  =  kernel capture timestamp of a frame  −  the draw time encoded in that frame
  D→A  =  input-receipt time encoded in the first frame that shows the new input count
          −  the serial-write timestamp of the command that caused it

Sync error is bounded by RTT/2 of the best sample and is reported per session, never assumed.
Modules: capture (V4L2, NV12 or MJPG), kmbox (serial HID, CP2102N/1a86 autodetect), probe (page + luminance
encoder/decoder), measure (instrument session), gymserve (gym frame server + page + input reports),
path (CaptureFrameSource and PhysicalRoute: the runner's two seams on hardware; calibration),
simtarget (the simulated target for --simulate runs and tests).
"""
