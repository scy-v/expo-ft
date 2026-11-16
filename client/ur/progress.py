"""In-place online action-budget display; no inference or control scheduling."""
import sys


class EpisodeProgress:
    def __init__(self, chunks, replan_steps, stream=None):
        self.chunks, self.replan_steps = chunks, replan_steps
        self.stream = sys.stderr if stream is None else stream
        self.active = False

    def update(self, episode, actions):
        completed, partial = divmod(actions, self.replan_steps)
        fraction = min(actions / (self.chunks * self.replan_steps), 1.)
        filled = int(20 * fraction)
        line = (f'Episode {episode} |[{"=" * filled}{" " * (20-filled)}]| '
                f'Chunks {completed}/{self.chunks} +{partial}/{self.replan_steps} actions '
                f'({actions}/{self.chunks*self.replan_steps})')
        if self.stream.isatty():
            self.stream.write('\r\033[2K' + line)
            self.stream.flush()
            self.active = True
        return line

    def close(self):
        if self.active:
            self.stream.write('\n')
            self.stream.flush()
            self.active = False
