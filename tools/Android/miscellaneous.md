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
        root.startActivity(intent);//from   w ww  .j  a  va2  s . c om
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