Title: Dotfiles with GNU Stow
Date: 22-05-2016
Category: tutorials

Once you start getting into Linux, you'll notice that most applications are
configurable through *dotfiles*; files like `.bashrc`, `.vimrc`, `.inputrc` et
cetera. These files normally reside inside your home directory, and this is
often the cause of severe file clutter.

There's multiple ways to make this mess a lot more manageable. I personally
use GNU Stow, a tool to manage symlinking packages of software/data.

At one point, my home directory looked approximately like this:

    /home/koen
    ├── .bashrc
    ├── .gitconfig
    ├── .i3
    ├── .tmux.conf
    ├── .vim
    ├── .vimrc
    ├── .Xresources
    ├── ...

A bunch of files and folders related to configuration of applications I use
daily. When your dotfiles are structured like this, you have practically no way
to use version control and/or keep track of every single file. Migrating your
configurations to another machine is even more of a hellish task.

Enter Stow and Git! You can create a single folder `~/dotfiles` that will
contain all of your configuration files and folders split up into packages.
My dotfiles folder looks like this (I left out the Git folders):

    /home/koen/dotfiles
    ├── bash
    │   ├── .bashrc
    │   └── .inputrc
    ├── git
    │   └── .gitconfig
    ├── i3
    │   └── .i3
    ├── powerline
    │   └── .config
    ├── README.md
    ├── tmux
    │   └── .tmux.conf
    ├── vim
    │   ├── .vim
    │   └── .vimrc
    └── x
        ├── .xinitrc
        └── .Xresources

The dotfiles folder is also a Git repository with a [remote on
Github](http://github.com/KDercksen/dotfiles/tree/desktop). I use Stow to
manage symlinking for me:

    :::sh
    cd ~/dotfiles
    stow bash
    stow vim
    stow i3
    ...

Stow will symlink to the contents of the specified folder from the home 
directory (you can also specify a target by using `-t`).

That's all there is to it! If anything is unclear, feel free to contact me.
