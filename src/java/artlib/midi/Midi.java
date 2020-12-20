package artlib.midi;

import javax.sound.midi.MidiEvent;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;
import java.util.ArrayList;
import java.util.List;

public class Midi {
    /**
     * @deprecated - can be re-written in pure clojure with a single `reduce` statement
     */
    public static Note[] getNotes(Track track) {
        List<Note> notes = new ArrayList<>();
        for (int i=0; i<track.size(); i++) {
            MidiEvent event = track.get(i);
            if (event.getMessage() instanceof ShortMessage && ((ShortMessage) event.getMessage()).getCommand() == ShortMessage.NOTE_ON && ((ShortMessage) event.getMessage()).getData2() > 0) {
                for (int j=i+1; j<track.size(); j++) {
                    MidiEvent event2 = track.get(j);
                    if (event2.getMessage() instanceof ShortMessage
                            && (((ShortMessage) event2.getMessage()).getCommand() == ShortMessage.NOTE_OFF || (((ShortMessage) event2.getMessage()).getCommand() == ShortMessage.NOTE_ON && ((ShortMessage) event2.getMessage()).getData2() == 0))
                            && (((ShortMessage) event2.getMessage()).getData1() == ((ShortMessage) event.getMessage()).getData1())) {
                        notes.add(new Note(((ShortMessage) event.getMessage()).getData1(), ((ShortMessage) event2.getMessage()).getData2(), (int) event.getTick(), (int) (event2.getTick() - event.getTick())));
                        break;
                    }
                }
            }
        }
        return notes.toArray(new Note[0]);
    }
}
