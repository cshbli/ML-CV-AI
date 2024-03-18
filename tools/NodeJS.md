# Node.js
* Node.js is an open source server environment.
* Node.js allows you to run JavaScript on the server.

## Installation

### Ubuntu

Installing via the PPA gives you a choice of version. Check out the available versions in the [documentation](https://github.com/nodesource/distributions/blob/master/README.md#installation-instructions). We'll also need curl to continue.
```
sudo apt install curl
```
Now we will download and run the script of the corresponding version, which will add the PPA to the system repositories.
```
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
```

And install Node.js from the added repositories.
```
sudo apt install nodejs
```

Let's check the installed version.
```
node -v
```
In this case, npm is already installed.

### Ubuntu Snap

- Enable snapd
```
sudo apt update
sudo apt install snapd
```

- Install Node
```
sudo snap install node --classic --channel=18
```

without specifying channel or using `--channel=20` may have issue:
```
/snap/node/8286/bin/node: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /snap/node/8286/bin/node)
```

### macOS
```
brew install node
```

## NPM
* NPM is a package manager for Node.js packages, or modules if you like.
* The NPM program is installed on your computer when you install Node.js

### Download a package
```
npm install <package-name>
```

### How to list all versions of an npm module?
```
npm view <package-name> versions
```
List the default version of an npm module
```
npm view <package-name> version
```

### Find the version of an installed npm package
Use `npm list` for local packages or `npm list -g` for globally installed packages.
You can find the version of a specific package by passing its name as an argument.
```
npm list
npm list <package-name>
```

## Node.js Application
https://www.w3schools.com/nodejs/nodejs_get_started.asp
