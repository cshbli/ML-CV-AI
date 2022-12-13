# Install PyQt5

```
pip install pyqt5
```

```
pip install pyqt5-tools
```
The installation will place the Qt Designer executable in a different directory according to your platform:

- Linux: ...lib/python3.x/site-packages/qt5_applications/Qt/bin/designer

for example: 
~/venv/pyqt/lib/python3.8/site-packages/qt5_applications/Qt/bin

```
~/venv/pyqt5/bin/pyuic5 -x demo.ui -o demo.py
```
