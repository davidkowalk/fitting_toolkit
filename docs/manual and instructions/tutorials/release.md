# Release Checklist

This document is a checklist for maintainers to release a new version of this package to PyPi.

## Quick Reference
- [ ] Make sure all tests are passing
- [ ] Build Package
- [ ] Locally Install Build
- [ ] Run examples to make sure package behaves as expected
- [ ] Merge onto stable
- [ ] Create a new Release Tag
- [ ] Publish to PyPi

## Building and Merging

For building the package reference the [related tutorial](./build_package.md). Before building for a release make sure you increment the version number in the `setup.py` and the `README.md` files. Then run:
```
python3 setup.py sdist bdist_wheel
```

## Publishing to PyPi

When the package is published to PyPi it can be installed via the 
```
pip install fitting-toolkit
```
command. For this maintainers have received an API token. This token is sensitive information and cannot be pushed. To ensure this the `.gitignore` includes the `pip_token` where it should be stored.

On upload PyPi clones all referenced images. This uncludes the shields. To ensure the correct version is displayed the package should first be published to GitHub, then to PyPi.

For uploading first make sure your `bin` folder only contains the filesto be uploaded. We publish both a wheel built and a source distribution. When in doubt delete your `bin` folder and rebuild. You can then publish the distribution with
```
twine upload ./dist/* -u __token__
``` 

You will then be asked for your token, which you can paste into the console. You will not see anything typed appear.
