project('xtsci-optimize', 'cpp',
  version : '0.0.1',
  default_options : ['warning_level=3',
                     'cpp_std=c++20',
                     'debug=true',
                     'buildtype=debug'])

# Add C++ compiler options
_args = [] # Extra arguments
_deps = [] # Dependencies
_linkto = [] # All the sub-libraries
_incdirs = [] # All the includes

add_languages('c', required: true)
cc = meson.get_compiler('c')
cppc = meson.get_compiler('cpp')
# Platform detection
host_system = host_machine.system()
is_windows = host_system == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

# Conditional arguments
if host_system == 'linux'
  _args += '-Wno-return-type'
  _args += '-Wno-switch'
  _args += '-Wno-unused-variable'
  _args += '-Wno-unused-const-variable'
endif

cpu_family = host_machine.cpu_family()

if is_mingw
  # For mingw-w64, don't use LTO
  add_project_arguments('-fno-use-linker-plugin', language: ['c', 'cpp'])
endif

# --------------------- Dependencies
if not is_windows
  # For building with clang
  _deps += [declare_dependency(link_args: '-lstdc++')]
endif

if cppc.get_id() == 'msvc'
  # Don't depend on VCRUNTIME140_1.dll
  # https://cibuildwheel.readthedocs.io/en/stable/faq/#windows-importerror-dll-load-failed-the-specific-module-could-not-be-found
  add_project_arguments('/d2FH4-', language : ['cpp', 'c'])
  if get_option('with_stdcall')
    add_project_arguments(cppc.get_supported_arguments('/Gz'), language : ['cpp', 'c'])
  endif
endif

# --------------------- Deps
_deps += dependency('fmt')
_deps += dependency('xtensor')
_deps += dependency('xtensor-blas')
_deps += [dependency('zlib'), dependency('xtensor-io')]

# --------------------- Subprojects
xtensor_fmt_proj = subproject('xtensor-fmt')
xtensor_fmt_dep = xtensor_fmt_proj.get_variable('xtensor_fmt_dep')
_deps += [xtensor_fmt_dep]


xtsci_function_proj = subproject('xtsci-function')
xtsci_function_dep = xtsci_function_proj.get_variable('xtsci_function_dep')
_deps += [xtsci_function_dep]

# --------------------- Projects
subdir('CppCore')
_incdirs += [ include_directories('CppCore') ]
xtsci_optimize_dep = declare_dependency(include_directories: _incdirs,
                                      link_with: _linkto,
                                      dependencies: _deps)
