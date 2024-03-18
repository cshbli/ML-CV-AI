# Flask

* Flask is a web application framework written in Python. 
* Flask is often referred to as a micro framework.  
* <b>Web Server Gateway Interface (WSGI)</b> has been adopted as a standard for Python web application development. WSGI is a specification for a universal interface between the web server and the web applications.
* <b>Werkzeug</b>: It is a WSGI toolkit, which implements requests, response objects, and other utility functions. This enables building a web framework on top of it. The Flask framework uses Werkzeug as one of its bases.
* <b>Jinja2</b> is a popular templating engine for Python. A web templating system combines a template with a certain data source to render dynamic web pages.

## Installation
* Install virtualenv for development environment
```
pip install virtualenv
```

* Create a virtual environment and activate it
```
python -m venv ~/venv/flask
source ~/venv/flask/bin/activate
```

* Install Flask in the virtual environment
```
pip install Flask
```

## Flask Application
https://www.tutorialspoint.com/flask/flask_application.htm