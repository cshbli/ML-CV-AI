# Miscellaneous

## Alert Dialog

```
// setup the alert builder
AlertDialog.Builder builder = new AlertDialog.Builder(this);
builder.setTitle("Delete");
builder.setMessage("Are you sure to delete?");

builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
    @Override
    public void onClick(DialogInterface dialog, int which) {
        // Do something
        

    }});
builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
    @Override
    public void onClick(DialogInterface dialog, int which) {

        // cancel
        dialog.cancel();
    }});

// create and show the alert dialog
AlertDialog dialog = builder.create();
dialog.show();
```

## Start Activity and Close Current

```
import android.app.Activity;
import android.content.Intent;

import android.os.Bundle;

public class Main {
    public static void startActivityAndCloseCurrent(Activity root,
            Class<?> activityClass) {
        Intent intent = new Intent(root, activityClass);
        root.startActivity(intent);
        root.finish();
    }

    public static void startActivity(Activity root, Class<?> activityClass) {
        Intent intent = new Intent(root, activityClass);
        root.startActivity(intent);
    }

    public static void startActivity(Activity root, Class<?> activityClass,
            Bundle extras) {
        if (extras == null) {
            throw new RuntimeException(
                    "Bundle is null, use method without bundle.");
        }

        Intent intent = new Intent(root, activityClass);
        intent.putExtras(extras);
        root.startActivity(intent);
    }
}
```

## Storage

- Internal Storage : Sensitive data, No other application access it.
- External Storage : Other application can access it like Images.

## Change the color of Action Bar in an Android App

[How to change the color of Action Bar in an Android App](https://www.geeksforgeeks.org/how-to-change-the-color-of-action-bar-in-an-android-app/)

## Change the background color of Button

```
android:backgroundTint
```

## Center text horizontally and vertically

Vertically
```
android:gravity="center"
```

Horizontally
```
android:textAlignment="center"
```

## Better way to define onClick

```
boilingpointK = (TextView) findViewById(R.id.boilingpointK);

boilingpointK.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        if ("Boiling Point K".equals(boilingpointK.getText().toString()))
            boilingpointK.setText("2792");
        else if ("2792".equals(boilingpointK.getText().toString()))
            boilingpointK.setText("Boiling Point K");
    }
});
```

## How to remove or set transparent background in icon

[How to remove or set transparent background in icon](https://stackoverflow.com/questions/8863633/how-to-remove-or-set-transparent-background-in-icon-jpg)

## Add border to a layout

### Define a drawable in `res/drawable` named as `custom_border.xml`
```
<?xml version="1.0" encoding="UTF-8"?>
 <shape xmlns:android="http://schemas.android.com/apk/res/android" android:shape="rectangle">
   <corners android:radius="5dp"/>
   <padding android:left="2dp" android:right="2dp" android:top="2dp" android:bottom="6dp"/>
   <stroke android:width="1dp" android:color="#CCCCCC"/>
 </shape>
```

### Set the background as the `custom_border`
```
<LinearLayout
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginStart="4dp"
        android:layout_marginTop="52dp"
        android:background="@drawable/custom_border"
        android:orientation="horizontal"      

        
    </LinearLayout>
```