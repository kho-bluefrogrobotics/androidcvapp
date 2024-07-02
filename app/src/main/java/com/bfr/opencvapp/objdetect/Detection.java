package com.bfr.opencvapp.objdetect;

import android.graphics.RectF;

// return object by tflite interpreter
public class Detection {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /**
     * Display name for the recognition.
     */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    public final Float confidence;

    /**
     * Optional location within the source image for the location of the recognized object.
     */
    private RectF location;

    private int detectedClass;

    public Detection(
            final String id, final String title, final Float confidence, final RectF location) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
    }

    public Detection(final String id, final String title, final Float confidence, final RectF location, int detectedClass) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
        this.detectedClass = detectedClass;
    }

    public Detection(final String id, final String title, final Float confidence, float left, float right, float top, float bottom, int detectedClass) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
        this.detectedClass = detectedClass;

        this.left= left;
        this.right=right;
        this.top=top;
        this.bottom=bottom;
    }

    public float left, right, top, bottom=0;

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public RectF getLocation() {
        return new RectF(location);
    }

    public void setLocation(RectF location) {
        this.location = location;
    }

    public int getDetectedClass() {
        return detectedClass;
    }

    public void setDetectedClass(int detectedClass) {
        this.detectedClass = detectedClass;
    }

    @Override
    public String toString() {
        String resultString = "";
        if (id != null) {
            resultString += "[" + id + "] ";
        }

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f);
        }

        if (location != null) {
            resultString += location + " ";
        }

        return resultString.trim();
    }
}