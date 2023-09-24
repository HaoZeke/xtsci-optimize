_incdirs += include_directories(['include'])

# --------------------- Executable

tiny_cli = executable('tiny_cli', 'tiny_cli.cpp',
                       dependencies: _deps,
                       include_directories: _incdirs,
                       cpp_args: _args,
                       link_with: _linkto,
                       install: false)

if get_option('with_tests')
    test_deps = _deps
    test_deps += dependency(
      'Catch2',
      method: 'cmake',
      modules: ['Catch2::Catch2WithMain'],
    )
    test_args = _args
    test_array = [#
      ['test_armijo', 'test_armijo.cc', ''],
      ['test_curvature', 'test_curvature.cc', ''],
      ['test_wolfe', 'test_wolfe.cc', ''],
      ['test_trial_funcs', 'test_trial_funcs.cc', ''],
      ['test_backtracking', 'test_backtracking.cc', ''],
      ['test_optim_cg', 'test_optim_cg.cc', ''],
    ]
    foreach test : test_array
      test(test.get(0),
           executable(test.get(0),
              sources : ['tests/'+test.get(1)],
              dependencies : test_deps,
              include_directories: _incdirs,
              cpp_args: test_args,
              link_with: _linkto,
                     ),
            workdir : meson.source_root() + test.get(2)
          )
    endforeach
endif
