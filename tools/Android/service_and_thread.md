
## Service vs Activity

- Services are not simply for running tasks in the background. They are for running tasks with a lifecyle that is independent of the Activity lifecycle (IE, they may continue after the activity is closed).

- A Service that starts and ends when an Activity starts and ends when the Activity ends is useless.

- A blocking operation is a Service will still block the application.

## Service vs Thread

- A Service is meant to run your task independently of the Activity, it allows you to run any task in background. This run on the main UI thread so when you want to perform any network or heavy load operation then you have to use the Thread there.

- A Threads is for run your task in its own thread instead of main UI thread. You would use when you want to do some heavy network operation like sending bytes to the server continuously, and it is associated with the Android components. When your component destroy who started this then you should have stop it also.

    Example : You are using the Thread in the Activity for some purpose, it is good practice to stop it when your activity destroy.

- Thread will sleep if your device sleeps. Whereas, Service can perform operation even if the device goes to sleep.

Use a Thread when: 

- app is required to be visible when the operation occurs.
- background operation is relatively short running (less than a minute or two)
- the activity/screen/app is highly coupled with the background operation, the user usually 'waits' for this operation to finish before doing anything else in the app. Using a thread in these cases leads to cleaner, more readable & maintainable code. That being said its possible to use a Service( or IntentService).

Use a Service when

- app could be invisible when the operation occurs
- User is not required to 'wait' for the operation to finish to do other things in the app.
- app is visible and the operation is independent of the app/screen context.