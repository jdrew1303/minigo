load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "38c3b21ea0dcf79bbc22d75f36fa57fb53ef2bf5f75e47f8b76af02c4a2abc1b",
    strip_prefix = "rules_closure-0.15.0",
    urls = [
        "https://github.com/bazelbuild/rules_closure/archive/0.15.0.tar.gz",
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz"],
)

http_file(
    name = "com_github_nlohmann_json_single_header",
    sha256 = "63da6d1f22b2a7bb9e4ff7d6b255cf691a161ff49532dcc45d398a53e295835f",
    urls = [
        "https://github.com/nlohmann/json/releases/download/v3.4.0/json.hpp",
    ],
)

http_archive(
    name = "libtensorflow",
    urls = ["https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz"],
    sha256 = "f0fd1eb5db9e4e0603f10aec289574d1decb54f73b675f0ce476fea1f05838c8",
    build_file_content = """
cc_library(
    name = "tensorflow_c",
    srcs = glob(["lib/libtensorflow.so*"]),
    hdrs = glob([
        "include/tensorflow/c/**/*.h",
        "include/tensorflow/core/platform/*.h",
        "include/tsl/**/*.h",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)

http_archive(
    name = "wtf",
    build_file = "//cc:wtf.BUILD",
    sha256 = "1837833cd159060f8bd6f6dd87edf854ed3135d07a6937b7e14b0efe70580d74",
    strip_prefix = "tracing-framework-fb639271fa3d56ed1372a792d74d257d4e0c235c",
    urls = ["https://github.com/google/tracing-framework/archive/fb639271fa3d56ed1372a792d74d257d4e0c235c.zip"],
)

http_archive(
    name = "com_google_absl",
    sha256 = "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed",
    strip_prefix = "abseil-cpp-20230802.1",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.1.tar.gz"],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)
