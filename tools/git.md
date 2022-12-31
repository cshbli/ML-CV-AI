# Git

## Store git personal access token

```
cat ~/.netrc

machine github.com login <login-id> password <token-password>
```

## Git running into a different owner than the repository

```
git config --global -add safe.directory <repo_path>
```

## Git to Ignore File Mode Changes

### Ignoring file mode changes
```
git config core.filemode false
```

Attaching the `--global` flag makes it a default for the logged user:
```
git config --global core.filemode false
```

### Caution
The `core.fileMode` is not recommended practice. It covers only the executable bit of mode not the read/write bits. There are cases when you use this setting just because you did `chmod -R 777` to make your files executable. But you should remember that most files shouldn’t be executable for many security reasons.

The appropriate way to solve this kind of problem is to manage the folder and file permission separately by running the following:

```
find . -type d -exec chmod a+rwx {} \; # Make folders traversable and read/write
find . -type f -exec chmod a+rw {} \;  # Make files read/write
```

### Configuration Levels

The `git config` command accepts arguments to define on which configuration level to operate. When searching for a configuration value, Git prioritizes the following orders:

--local

The git config command writes to a local level by default if no configuration option is passed. The repository of the `.git` directory has a file (config) which stores the local configuration values.

--global

The application of this level includes the operating system user. Global configuration values are found in a file (.gitconfig) located in a user's home directory.

--system

This configuration includes all users on an operating system and all repositories. The System-level configuration file is placed in the git config file of the system root path.
