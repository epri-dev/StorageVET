import PyInstaller.__main__
import os
import SVETapp

package_name = 'SVETapp'

PyInstaller.__main__.run([
    '--name=%s' % package_name,
    '--onedir',
    '--no-console',
    # '--add-binary=%s' % os.path.join('resource', 'path', '*.png'),
    # '--add-data=%s' % os.path.join('resource', 'path', '*.txt'),
    '--icon=%s' % os.path.join('resource', 'path', 'favicon.ico'),
    os.path.join('my_package', '__main__.py'),
])
