# Application Class

## Overview

The Application class in Android is the base class within an Android app that contains all other components such as activities and services. The Application class, or any subclass of the Application class, is instantiated before any other class when the process for your application/package is created.

<img src="figs/b4YiAfy.png">

This class is primarily used for initialization of global state before the first Activity is displayed. Note that custom Application objects should be used carefully and are often not needed at all.

## Custom Application Classes

In many apps, there's no need to work with an application class directly. However, there are a few acceptable uses of a custom application class:

- Specialized tasks that need to run before the creation of your first activity
- Global initialization that needs to be shared across all components (crash reporting, persistence)
- Static methods for easy access to static immutable data such as a shared network client object

Note that you should never store mutable shared data inside the Application object since that data might disappear or become invalid at any time. Instead, store any mutable shared data using persistence strategies such as files, SharedPreferences or SQLite.

## Defining Your Application Class

If we do want a custom application class, we start by creating a new class which extends `android.app.Application` as follows:

```
import android.app.Application;

public class MyCustomApplication extends Application {
        // Called when the application is starting, before any other application objects have been created.
        // Overriding this method is totally optional!
	@Override
	public void onCreate() {
	    super.onCreate();
            // Required initialization logic here!
	}

        // Called by the system when the device configuration changes while your component is running.
        // Overriding this method is totally optional!
	@Override
	public void onConfigurationChanged(Configuration newConfig) {
	    super.onConfigurationChanged(newConfig);
	}

        // This is called when the overall system is running low on memory, 
        // and would like actively running processes to tighten their belts.
        // Overriding this method is totally optional!
	@Override
	public void onLowMemory() {
	    super.onLowMemory();
	}
}
```

And specify the `android:name` property in the the <application> node in `AndroidManifest.xml`:

```
<application 
   android:name=".MyCustomApplication"
   android:icon="@drawable/icon" 
   android:label="@string/app_name" 
   ...>
```                       

## getApplication()

The `getApplication()` method is located in the `Activity` class. We can use `getApplication()` method from all activities.

Whenever you want `getApplication()` in a non activity class you have to pass an `Activity` instance to the constructor of that non activity class.

