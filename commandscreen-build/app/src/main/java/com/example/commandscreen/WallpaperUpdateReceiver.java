package com.example.commandscreen;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;

public class WallpaperUpdateReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        try {
            Bitmap bitmap = WallpaperEngine.render(context);
            WallpaperEngine.setLockWallpaper(context, bitmap);
        } catch (Throwable ignored) {
        }
    }
}
