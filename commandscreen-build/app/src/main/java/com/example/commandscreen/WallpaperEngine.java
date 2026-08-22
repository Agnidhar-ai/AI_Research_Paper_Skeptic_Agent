package com.example.commandscreen;

import android.app.AlarmManager;
import android.app.PendingIntent;
import android.app.WallpaperManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.util.DisplayMetrics;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Calendar;
import java.util.Locale;

final class WallpaperEngine {
    private WallpaperEngine() {}

    static final Event[] EVENTS = new Event[]{
            new Event("Engineering Mathematics", "09:00", "10:00", Color.rgb(255, 107, 107)),
            new Event("Defense Technology", "11:30", "12:30", Color.rgb(255, 209, 102)),
            new Event("Data Engineering Lab", "15:00", "16:00", Color.rgb(142, 202, 230)),
            new Event("Project Work", "19:00", "20:30", Color.rgb(205, 180, 219))
    };

    static String previewText() {
        StringBuilder sb = new StringBuilder();
        sb.append(LocalDate.now().format(DateTimeFormatter.ofPattern("EEE, MMM d", Locale.ENGLISH)).toUpperCase(Locale.ENGLISH));
        sb.append("\n\nMON   TUE   WED   THU   FRI   SAT   SUN\n\n");
        for (Event event : EVENTS) {
            sb.append("│ ").append(event.title).append("\n");
            sb.append("│ ").append(event.start).append(" – ").append(event.end).append("\n\n");
        }
        return sb.toString();
    }

    static Bitmap render(Context context) {
        DisplayMetrics metrics = context.getResources().getDisplayMetrics();
        int width = Math.max(metrics.widthPixels, 1080);
        int height = Math.max(metrics.heightPixels, 2400);
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);

        Paint background = new Paint(Paint.ANTI_ALIAS_FLAG);
        background.setShader(new LinearGradient(0, 0, width, height,
                new int[]{Color.rgb(55, 37, 33), Color.rgb(20, 15, 14), Color.rgb(43, 31, 27)},
                null, Shader.TileMode.CLAMP));
        canvas.drawRect(0, 0, width, height, background);

        Paint white = new Paint(Paint.ANTI_ALIAS_FLAG);
        white.setColor(Color.WHITE);
        Paint muted = new Paint(Paint.ANTI_ALIAS_FLAG);
        muted.setColor(Color.rgb(205, 197, 194));

        String date = LocalDate.now().format(DateTimeFormatter.ofPattern("EEE, MMM d", Locale.ENGLISH)).toUpperCase(Locale.ENGLISH);
        white.setTextSize(width * 0.052f);
        white.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        canvas.drawText(date, width * 0.08f, height * 0.31f, white);

        muted.setTextSize(width * 0.025f);
        muted.setTypeface(Typeface.DEFAULT);
        canvas.drawText("MON     TUE     WED     THU     FRI     SAT     SUN", width * 0.08f, height * 0.36f, muted);

        float y = height * 0.45f;
        for (Event event : EVENTS) {
            Paint bar = new Paint(Paint.ANTI_ALIAS_FLAG);
            bar.setColor(event.color);
            canvas.drawRoundRect(width * 0.08f, y - width * 0.035f, width * 0.09f, y + width * 0.075f, 12f, 12f, bar);

            white.setTextSize(width * 0.044f);
            white.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
            canvas.drawText(event.title, width * 0.12f, y, white);

            muted.setTextSize(width * 0.030f);
            muted.setTypeface(Typeface.DEFAULT);
            canvas.drawText(event.start + " – " + event.end, width * 0.12f, y + width * 0.048f, muted);
            y += height * 0.105f;
        }
        return bitmap;
    }

    static void setLockWallpaper(Context context, Bitmap bitmap) throws Exception {
        WallpaperManager.getInstance(context).setBitmap(bitmap, null, true, WallpaperManager.FLAG_LOCK);
    }

    static void scheduleDaily(Context context) {
        AlarmManager alarmManager = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        if (alarmManager == null) return;
        Intent intent = new Intent(context, WallpaperUpdateReceiver.class);
        PendingIntent pendingIntent = PendingIntent.getBroadcast(context, 630, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Calendar next = Calendar.getInstance();
        next.set(Calendar.HOUR_OF_DAY, 6);
        next.set(Calendar.MINUTE, 30);
        next.set(Calendar.SECOND, 0);
        next.set(Calendar.MILLISECOND, 0);
        if (next.getTimeInMillis() <= System.currentTimeMillis()) next.add(Calendar.DAY_OF_YEAR, 1);
        alarmManager.setInexactRepeating(AlarmManager.RTC_WAKEUP, next.getTimeInMillis(), AlarmManager.INTERVAL_DAY, pendingIntent);
    }

    static final class Event {
        final String title;
        final String start;
        final String end;
        final int color;
        Event(String title, String start, String end, int color) {
            this.title = title; this.start = start; this.end = end; this.color = color;
        }
    }
}
