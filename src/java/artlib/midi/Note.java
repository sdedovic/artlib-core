package artlib.midi;

public class Note {
    private final int data1;
    private final int data2;
    private final int trackTime;
    private final int duration;

    public Note(int data1, int data2, int trackTime, int duration) {
        this.data1 = data1;
        this.data2 = data2;
        this.trackTime = trackTime;
        this.duration = duration;
    }

    public int getData1() {
        return data1;
    }

    public int getData2() {
        return data2;
    }

    public int getTrackTime() {
        return trackTime;
    }

    public int getDuration() {
        return duration;
    }
}
