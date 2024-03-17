# Node.js

## Installation

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