window.BENCHMARK_DATA = {
  "lastUpdate": 1761908815048,
  "repoUrl": "https://github.com/joanjcaceres/HybridSuperQubits",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "73915906+joanjcaceres@users.noreply.github.com",
            "name": "Joan J. Cáceres",
            "username": "joanjcaceres"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "97c5c0728601784401bb31ff3d9b08024fcd884e",
          "message": "Update README.md",
          "timestamp": "2025-07-31T14:22:00+02:00",
          "tree_id": "120f2fdd51e435628f24498ad64c5a6333597865",
          "url": "https://github.com/joanjcaceres/HybridSuperQubits/commit/97c5c0728601784401bb31ff3d9b08024fcd884e"
        },
        "date": 1753964840376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_hamiltonian_calculation_speed",
            "value": 41.946043595091595,
            "unit": "iter/sec",
            "range": "stddev: 0.03601113760518261",
            "extra": "mean: 23.84015068627395 msec\nrounds: 51"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_eigenvalue_calculation_speed",
            "value": 34.745115752744866,
            "unit": "iter/sec",
            "range": "stddev: 0.029332201554505156",
            "extra": "mean: 28.781023701755835 msec\nrounds: 57"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_eigensystem_calculation_speed",
            "value": 35.75256382529289,
            "unit": "iter/sec",
            "range": "stddev: 0.02152119829289446",
            "extra": "mean: 27.970022090906873 msec\nrounds: 22"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_parameter_sweep_speed",
            "value": 1.6881772113598512,
            "unit": "iter/sec",
            "range": "stddev: 0.30886830739121157",
            "extra": "mean: 592.3548743999959 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_matrix_elements_speed",
            "value": 3.7862832830020765,
            "unit": "iter/sec",
            "range": "stddev: 0.15495214243903155",
            "extra": "mean: 264.11124716667206 msec\nrounds: 6"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_hamiltonian_calculation_speed",
            "value": 2.8608155017897134,
            "unit": "iter/sec",
            "range": "stddev: 0.09181167186332082",
            "extra": "mean: 349.55067860000213 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_eigenvalue_calculation_speed",
            "value": 2.2203972175516045,
            "unit": "iter/sec",
            "range": "stddev: 0.11230444387953478",
            "extra": "mean: 450.3698672000155 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_jrl_potential_speed",
            "value": 5.326394129787078,
            "unit": "iter/sec",
            "range": "stddev: 0.06437107538027642",
            "extra": "mean: 187.7442741999971 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_reduced_density_matrix_speed",
            "value": 2.3063960421505962,
            "unit": "iter/sec",
            "range": "stddev: 0.07236512574410708",
            "extra": "mean: 433.57687999999825 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestOperatorBenchmarks::test_n_operator_speed",
            "value": 24109.811638283598,
            "unit": "iter/sec",
            "range": "stddev: 0.000009035191821996574",
            "extra": "mean: 41.476889782586085 usec\nrounds: 8329"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestOperatorBenchmarks::test_phase_operator_speed",
            "value": 35831.58659375474,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027282959295169693",
            "extra": "mean: 27.908337170151846 usec\nrounds: 15986"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[20]",
            "value": 3413.8554868433553,
            "unit": "iter/sec",
            "range": "stddev: 0.0003083232693956457",
            "extra": "mean: 292.92394005952985 usec\nrounds: 2002"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[30]",
            "value": 2303.4745792521367,
            "unit": "iter/sec",
            "range": "stddev: 0.00004273169398032701",
            "extra": "mean: 434.1267791740369 usec\nrounds: 1671"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[40]",
            "value": 1475.9004959681552,
            "unit": "iter/sec",
            "range": "stddev: 0.00012574006176425037",
            "extra": "mean: 677.5524520330378 usec\nrounds: 1230"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[50]",
            "value": 37.07302420858856,
            "unit": "iter/sec",
            "range": "stddev: 0.02871791311099761",
            "extra": "mean: 26.973790818185638 msec\nrounds: 22"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[5]",
            "value": 15.634224870008687,
            "unit": "iter/sec",
            "range": "stddev: 0.04799810537757098",
            "extra": "mean: 63.962237227271274 msec\nrounds: 22"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[10]",
            "value": 14.737343956572454,
            "unit": "iter/sec",
            "range": "stddev: 0.05257498224355362",
            "extra": "mean: 67.85483211539128 msec\nrounds: 26"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[15]",
            "value": 20.3197640092931,
            "unit": "iter/sec",
            "range": "stddev: 0.04583174102973426",
            "extra": "mean: 49.213169972971 msec\nrounds: 37"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[20]",
            "value": 17.436697706613355,
            "unit": "iter/sec",
            "range": "stddev: 0.04510213459466135",
            "extra": "mean: 57.35030891891428 msec\nrounds: 37"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[5]",
            "value": 3.926865443427021,
            "unit": "iter/sec",
            "range": "stddev: 0.14383143836977774",
            "extra": "mean: 254.65603912500967 msec\nrounds: 8"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[10]",
            "value": 1.6145755375235693,
            "unit": "iter/sec",
            "range": "stddev: 0.11266619921607475",
            "extra": "mean: 619.357829199987 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[20]",
            "value": 1.2463207424562246,
            "unit": "iter/sec",
            "range": "stddev: 0.3029784005148196",
            "extra": "mean: 802.361676199996 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "73915906+joanjcaceres@users.noreply.github.com",
            "name": "Joan J. Cáceres",
            "username": "joanjcaceres"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fc45f4819854f13b83f8ad86b7aede1d375eb12d",
          "message": "Update README.md",
          "timestamp": "2025-07-31T14:22:53+02:00",
          "tree_id": "2783031ef0e8ea0326f49bc57984bdff52a520e0",
          "url": "https://github.com/joanjcaceres/HybridSuperQubits/commit/fc45f4819854f13b83f8ad86b7aede1d375eb12d"
        },
        "date": 1753965021525,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_hamiltonian_calculation_speed",
            "value": 18.76505332577802,
            "unit": "iter/sec",
            "range": "stddev: 0.056764315834883845",
            "extra": "mean: 53.290549333333104 msec\nrounds: 9"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_eigenvalue_calculation_speed",
            "value": 17.667933137114424,
            "unit": "iter/sec",
            "range": "stddev: 0.054944509305663906",
            "extra": "mean: 56.599716120689536 msec\nrounds: 58"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_eigensystem_calculation_speed",
            "value": 24.61872041412626,
            "unit": "iter/sec",
            "range": "stddev: 0.028868722976228996",
            "extra": "mean: 40.619495374999204 msec\nrounds: 8"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_parameter_sweep_speed",
            "value": 1.7883914508709338,
            "unit": "iter/sec",
            "range": "stddev: 0.09841982346452949",
            "extra": "mean: 559.1616978000019 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_matrix_elements_speed",
            "value": 2.704414651996632,
            "unit": "iter/sec",
            "range": "stddev: 0.19938334962560264",
            "extra": "mean: 369.7657825 msec\nrounds: 6"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_hamiltonian_calculation_speed",
            "value": 3.1550953502628114,
            "unit": "iter/sec",
            "range": "stddev: 0.030719602271604425",
            "extra": "mean: 316.94763199999727 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_eigenvalue_calculation_speed",
            "value": 2.771926160036136,
            "unit": "iter/sec",
            "range": "stddev: 0.03769205067173322",
            "extra": "mean: 360.75997059999736 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_jrl_potential_speed",
            "value": 5.923649178572758,
            "unit": "iter/sec",
            "range": "stddev: 0.05400875048816743",
            "extra": "mean: 168.81485885714451 msec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_reduced_density_matrix_speed",
            "value": 2.546096727341059,
            "unit": "iter/sec",
            "range": "stddev: 0.13570878577027462",
            "extra": "mean: 392.75805560000094 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestOperatorBenchmarks::test_n_operator_speed",
            "value": 24369.79188171229,
            "unit": "iter/sec",
            "range": "stddev: 0.000008718315570784715",
            "extra": "mean: 41.03440869966663 usec\nrounds: 8598"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestOperatorBenchmarks::test_phase_operator_speed",
            "value": 35715.2737238718,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027874419191632532",
            "extra": "mean: 27.999225421912648 usec\nrounds: 17536"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[20]",
            "value": 3716.0221150755465,
            "unit": "iter/sec",
            "range": "stddev: 0.000032060543440336854",
            "extra": "mean: 269.10496467259856 usec\nrounds: 2123"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[30]",
            "value": 2302.843184428837,
            "unit": "iter/sec",
            "range": "stddev: 0.0001464054589683255",
            "extra": "mean: 434.24580829546375 usec\nrounds: 1977"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[40]",
            "value": 1500.4940195267902,
            "unit": "iter/sec",
            "range": "stddev: 0.000012689375750343504",
            "extra": "mean: 666.4471747214089 usec\nrounds: 538"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[50]",
            "value": 20.796815100866358,
            "unit": "iter/sec",
            "range": "stddev: 0.02335870620385046",
            "extra": "mean: 48.08428575000129 msec\nrounds: 28"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[5]",
            "value": 11.613458676662596,
            "unit": "iter/sec",
            "range": "stddev: 0.03646873581273348",
            "extra": "mean: 86.1069925714304 msec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[10]",
            "value": 9.518779501559054,
            "unit": "iter/sec",
            "range": "stddev: 0.03612000404940339",
            "extra": "mean: 105.05548530000226 msec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[15]",
            "value": 10.480451175997041,
            "unit": "iter/sec",
            "range": "stddev: 0.05001971397045674",
            "extra": "mean: 95.415739571428 msec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[20]",
            "value": 10.774360917640678,
            "unit": "iter/sec",
            "range": "stddev: 0.04046708880902477",
            "extra": "mean: 92.81292947618981 msec\nrounds: 21"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[5]",
            "value": 3.0544941146083375,
            "unit": "iter/sec",
            "range": "stddev: 0.11407183004945684",
            "extra": "mean: 327.38645500000416 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[10]",
            "value": 1.5763681127578273,
            "unit": "iter/sec",
            "range": "stddev: 0.04872080397127539",
            "extra": "mean: 634.3695941999982 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[20]",
            "value": 1.1079991814093215,
            "unit": "iter/sec",
            "range": "stddev: 0.3387720414439219",
            "extra": "mean: 902.5277425999974 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "joan.caceres@ib.edu.ar",
            "name": "Joan Caceres",
            "username": "joanjcaceres"
          },
          "committer": {
            "email": "joan.caceres@ib.edu.ar",
            "name": "Joan Caceres",
            "username": "joanjcaceres"
          },
          "distinct": true,
          "id": "60a1ef52eeba0720e8492de366ed348702d73d05",
          "message": "Fix: Ensure directory creation for file write in SpectrumData class",
          "timestamp": "2025-10-31T11:59:57+01:00",
          "tree_id": "23d793959939fccc522ba9d05f448177608225c3",
          "url": "https://github.com/joanjcaceres/HybridSuperQubits/commit/60a1ef52eeba0720e8492de366ed348702d73d05"
        },
        "date": 1761908814563,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_hamiltonian_calculation_speed",
            "value": 9.890152646000582,
            "unit": "iter/sec",
            "range": "stddev: 0.04698265720366255",
            "extra": "mean: 101.11067399999978 msec\nrounds: 6"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_eigenvalue_calculation_speed",
            "value": 19.2272004127887,
            "unit": "iter/sec",
            "range": "stddev: 0.023347934988538373",
            "extra": "mean: 52.00965187500017 msec\nrounds: 24"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_eigensystem_calculation_speed",
            "value": 18.16792595336542,
            "unit": "iter/sec",
            "range": "stddev: 0.037326806269359664",
            "extra": "mean: 55.042056125000904 msec\nrounds: 16"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_parameter_sweep_speed",
            "value": 1.593471098149787,
            "unit": "iter/sec",
            "range": "stddev: 0.12251232908286677",
            "extra": "mean: 627.5608017999957 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFluxoniumBenchmarks::test_matrix_elements_speed",
            "value": 2.522671571049629,
            "unit": "iter/sec",
            "range": "stddev: 0.11476084591377946",
            "extra": "mean: 396.4051490000031 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_hamiltonian_calculation_speed",
            "value": 9.319167397441216,
            "unit": "iter/sec",
            "range": "stddev: 0.09430335721909344",
            "extra": "mean: 107.30572349999552 msec\nrounds: 6"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_eigenvalue_calculation_speed",
            "value": 3.4916147962848467,
            "unit": "iter/sec",
            "range": "stddev: 0.1848231139961559",
            "extra": "mean: 286.40043599999103 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_jrl_potential_speed",
            "value": 4.442596840346376,
            "unit": "iter/sec",
            "range": "stddev: 0.031769657887478035",
            "extra": "mean: 225.09357385713915 msec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestFerboBenchmarks::test_reduced_density_matrix_speed",
            "value": 2.3878846454553395,
            "unit": "iter/sec",
            "range": "stddev: 0.06882871158641635",
            "extra": "mean: 418.7806985999998 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestOperatorBenchmarks::test_n_operator_speed",
            "value": 23155.07588264304,
            "unit": "iter/sec",
            "range": "stddev: 0.00010913848066381134",
            "extra": "mean: 43.18707505293025 usec\nrounds: 11312"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestOperatorBenchmarks::test_phase_operator_speed",
            "value": 35806.73431593603,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032540389048971487",
            "extra": "mean: 27.92770743002227 usec\nrounds: 16191"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[20]",
            "value": 3715.7084043226682,
            "unit": "iter/sec",
            "range": "stddev: 0.00002035108564223114",
            "extra": "mean: 269.12768473345494 usec\nrounds: 2214"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[30]",
            "value": 2319.850816937122,
            "unit": "iter/sec",
            "range": "stddev: 0.00006999234983301084",
            "extra": "mean: 431.0622013704704 usec\nrounds: 1897"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[40]",
            "value": 1487.903074832161,
            "unit": "iter/sec",
            "range": "stddev: 0.00002555761713580884",
            "extra": "mean: 672.0867890623872 usec\nrounds: 1280"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_fluxonium_scaling_with_dimension[50]",
            "value": 42.69646413469313,
            "unit": "iter/sec",
            "range": "stddev: 0.01578106344537226",
            "extra": "mean: 23.42114318519053 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[5]",
            "value": 24.782120245452866,
            "unit": "iter/sec",
            "range": "stddev: 0.044946307881326156",
            "extra": "mean: 40.351672499994606 msec\nrounds: 8"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[10]",
            "value": 8.638352245780824,
            "unit": "iter/sec",
            "range": "stddev: 0.041275494542499465",
            "extra": "mean: 115.76281813333367 msec\nrounds: 15"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[15]",
            "value": 13.932336354583784,
            "unit": "iter/sec",
            "range": "stddev: 0.034010085212322655",
            "extra": "mean: 71.77547071428523 msec\nrounds: 21"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_eigenvalue_count_scaling[20]",
            "value": 12.26643884591396,
            "unit": "iter/sec",
            "range": "stddev: 0.041115324041190296",
            "extra": "mean: 81.52325320833498 msec\nrounds: 24"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[5]",
            "value": 2.2665501015609784,
            "unit": "iter/sec",
            "range": "stddev: 0.10650958633092339",
            "extra": "mean: 441.1991596000007 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[10]",
            "value": 1.4631701202902052,
            "unit": "iter/sec",
            "range": "stddev: 0.12634468734453771",
            "extra": "mean: 683.4475268000006 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestScalingBenchmarks::test_parameter_sweep_scaling[20]",
            "value": 0.6515670873116551,
            "unit": "iter/sec",
            "range": "stddev: 0.2028487552045386",
            "extra": "mean: 1.5347613767999975 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}