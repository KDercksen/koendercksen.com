Title: Python project directory structure
Date: 25-07-2016
Category: tutorials

I have struggled with project directory structure for Python projects in the
past. Some things (like relative imports in projects containing multiple
modules in different folders) are not immediately clear to most people, which
is why I'm posting a concise little write-up of my knowledge on the topic so
far. I have created a [Github
repository](http://github.com/KDercksen/pydirstructure) with an example project
to assist me in explaining some things; I'll be referring to that regularly
throughout this post.

## Structuring your project

Let's start off with the full directory structure:

    pydirstructure/
    ├── LICENSE.md
    ├── pydirstructure
    │   ├── __init__.py
    │   ├── module_a.py
    │   ├── module_b.py
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    └── tests
        └── test_a.py

The name of the root directory is not relevant, but I almost always name it
identically to the package name. The directory containing the actual code for
the package does have to have a relevant name (since it'll be used in `import
<package_name>`), hence it's called `pydirstructure`, the name of our package.

Most of the files located in the root folder are not going to be very important
to your project itself, since they're often configuration files for CI hooks,
examples and other stuff. However, `setup.py` (and `requirements.txt` in a
sense) are very important to your Python project.

### requirements.txt

`requirements.txt` contains the dependencies for your project and can be
populated using `pip freeze > requirements.txt`.

### setup.py

`setup.py` actually contains information regarding installing, testing and
deploying your package. Most of it should speak for itself, but I'll go into
three little handy things.

#### 1. Parsing requirements.txt

You can see (on line 9 in `setup.py`) that I'm parsing the requirements file
instead of adding all requirements manually to the `setup` functions keyword
arguments. `parse_requirements` gives a list of installation requirement objects,
which can be converted to strings using `str`.

#### 2. Versioning in toplevel \_\_init\_\_.py

I prefer to keep track of the package version in `package/__init__.py` and
refer to that version from all other places. This allows me to keep the version
string in only one place (I'd forget to update all of them, guaranteed, if I
had more than one).

#### 3. Entry points

I often write projects that offer commandline interfaces as part of their
functionality.  You can specify those in `setup` using the `entry_points`
parameter `console_scripts`. The syntax is pretty simple:

    cli_script_name=package.module:function

So if we make the main function in pydirstructure/module_a.py a script:

    cli_script_name=pydirstructure.module_a:main

Easy enough.

That's a simple example of a `setup.py` that you can use to start out. Of
course you can do way more complicated things, but I'd still be typing in 5
years.  There's plenty of documentation online!

### package/\_\_init\_\_.py

This is the file that glues your `package` module together. It makes things
like this possible:

    :::python
    import pydirstructure as pds


    s1 = pds.SomeClass().do_something('waddup readers :D')
    s2 = pds.SomeOtherClass().do_something_else('waddup yalllll')

    print(s1)
    print(s2)

Even though `SomeClass` is in `module_a` and `SomeOtherClass` is in `module_b`.
A package can have multiple `__init__.py` files; there should be one in every
module directory (so if we had more directories below the package directory, 
each of those should have an `__init__.py` to be recognized as a module).

## Editable installation with pip

Due to relative imports, you can't simply make each Python file in a structured
package runnable and expect to be able to run all the different files on their
own. However, once you get your `setup.py` up and running, you can use the
following `pip` command in the project directory to install it *editable*:

    pip install -e .

(protip: work in a virtual environment if you're going to do this, which you
should already be doing anyway).
