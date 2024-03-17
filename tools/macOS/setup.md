# macOS setup

## .zshrc

```
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools/:$ANDROID_HOME/emulator/:

alias python='python3'
alias pip='pip3'

PS1="%{%F{033}%}%n%{%f%}@%{%F{green}%}%m:%{%F{yellow}%}%~%{$%f%}%  "
export CLICOLOR=1
export LSCOLORS=ExFxBxDxCxegedabagacad
```
## Install Visual Studio Code
## Install Xcode
## Install Android Studio
## Install JDK
Follow Oracle's official JDK Installation Guide, instead of `brew`. 
## Install homebrew
https://stackoverflow.com/questions/66666134/how-to-install-homebrew-on-m1-mac
## Install Node
```
brew install node
```
