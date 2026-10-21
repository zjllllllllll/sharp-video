// Every source frame is presented in order. The clock never jumps to a
// wall-time-derived frame index when the main thread is briefly late.
export class FrameClock {
  constructor(count, fps) { this.count = count; this.interval = 1000 / fps; this.frame = 0; this.playing = false; this.due = 0; }
  play(now) { this.playing = true; this.due = now + this.interval; }
  pause() { this.playing = false; }
  seek(frame) { this.frame = Math.max(0, Math.min(this.count - 1, frame)); }
  tick(now) {
    if (!this.playing || now < this.due) return null;
    this.frame = (this.frame + 1) % this.count;
    this.due += this.interval;
    // A long suspension changes duration rather than deleting source frames.
    if (this.due < now - this.interval) this.due = now + this.interval;
    return this.frame;
  }
}
