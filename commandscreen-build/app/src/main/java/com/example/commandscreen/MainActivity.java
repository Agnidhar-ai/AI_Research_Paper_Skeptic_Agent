package com.example.commandscreen;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public class MainActivity extends Activity {
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WallpaperEngine.scheduleDaily(this);
        setContentView(buildUi());
    }

    private ScrollView buildUi() {
        int pad = dp(24);
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(Color.rgb(22, 16, 15));

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        scroll.addView(root, new ScrollView.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = text("Command Screen V2", 30, Color.WHITE);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        root.addView(title);

        TextView subtitle = text("Phase 0.1 • sample schedule → lock-screen wallpaper", 15, Color.rgb(190, 182, 178));
        subtitle.setPadding(0, dp(8), 0, dp(22));
        root.addView(subtitle);

        TextView preview = text(WallpaperEngine.previewText(), 17, Color.WHITE);
        preview.setBackgroundColor(Color.rgb(40, 29, 27));
        preview.setPadding(dp(18), dp(20), dp(18), dp(20));
        preview.setLineSpacing(0, 1.3f);
        root.addView(preview, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        Button setButton = new Button(this);
        setButton.setText("Generate & Set Lock Screen");
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(58));
        buttonParams.setMargins(0, dp(24), 0, 0);
        root.addView(setButton, buttonParams);

        Button rescheduleButton = new Button(this);
        rescheduleButton.setText("Refresh Daily Schedule");
        LinearLayout.LayoutParams secondaryParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(54));
        secondaryParams.setMargins(0, dp(10), 0, 0);
        root.addView(rescheduleButton, secondaryParams);

        status = text("Ready • daily refresh scheduled around 06:30", 14, Color.rgb(190, 182, 178));
        status.setGravity(Gravity.CENTER_HORIZONTAL);
        status.setPadding(0, dp(18), 0, dp(20));
        root.addView(status);

        setButton.setOnClickListener(v -> {
            try {
                Bitmap bitmap = WallpaperEngine.render(this);
                WallpaperEngine.setLockWallpaper(this, bitmap);
                status.setText("✓ Lock-screen wallpaper updated");
            } catch (Throwable t) {
                status.setText("Could not set wallpaper: " + t.getClass().getSimpleName());
            }
        });

        rescheduleButton.setOnClickListener(v -> {
            WallpaperEngine.scheduleDaily(this);
            status.setText("✓ Daily refresh rescheduled for ~06:30");
        });

        return scroll;
    }

    private TextView text(String value, int sp, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(sp);
        view.setTextColor(color);
        return view;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
