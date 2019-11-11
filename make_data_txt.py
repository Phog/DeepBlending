
# Copyright 2019 Peter Hedman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import os
from PIL import Image

if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("Usage: ./" + sys.argv[0] + " [PATH_TO_DATASET_FOLDER] (validation/test)")
    sys.exit(1)

is_validation = len(sys.argv) > 2 and sys.argv[2] == "validation"
is_test       = len(sys.argv) > 2 and sys.argv[2] == "test"

dataset_path    = sys.argv[1]
dataset_dirname = os.path.split(dataset_path)[-1]

files = os.listdir(dataset_path)
if len(files) <= 0:
    print("ERROR: No files found in " + dataset_path)
    sys.exit(1)

indices = sorted(list(set([f.split("_")[0] for f in files])))

first_image_files = sorted([f for f in files if str(indices[0]) == f[0:len(indices[0])]])
first_image_index = first_image_files[0].split("_")[0]

if not is_test:
    reference_file      = [f for f in first_image_files if "reference" in f][0]
    reference_suffix    = "_reference"
    reference_extension = reference_file[-4:]

probe_filename   = [f for f in files if "global" in f][0]
has_path_samples = probe_filename.find("path") >= 0
random_suffixes  = [""]
path_suffixes    = [""]
if has_path_samples:
    num_filename_chunks = len(probe_filename.split("_"))
    if num_filename_chunks != 7:
        print("ERROR: Expected file name to contain seven chunks, found: " + str(num_filename_chunks))
        print("(" + probe_filename + ")")
        sys.exit(0)

    flow_file      = [f for f in first_image_files if "temporal_flow" in f][0]
    flow_suffix    = "_temporal_flow"
    flow_extension = flow_file[-4:]

    # Build suffixes of the form _sample_000N
    random_sample_files = [f for f in first_image_files if "global" in f and "path_0000" in f]
    random_suffixes     = sorted(list(set(["_sample_" + f.split("_")[-3] for f in random_sample_files])))

    # Build suffixes of the form _path_000N
    path_sample_files  = [f for f in first_image_files if "global" in f and "sample_0000" in f]
    path_suffixes      = sorted(list(set(["_path_" + f.split("_")[-1][0:-4] for f in path_sample_files])))

    # Find the remaining suffixes: _local_layer_N and _global_colors
    first_sample_files = [f for f in first_image_files if random_suffixes[0] in f and path_suffixes[0] in f]
    type_files         = [f for f in first_sample_files if "reference" not in f and "temporal_flow" not in f]
    type_suffixes      = ["_" + "_".join(f.split("_")[1:-4]) for f in type_files]
    type_extensions    = [f.split("_")[-1][-4:] for f in type_files]
else:
    type_files      = [f for f in first_image_files if "reference" not in f]
    type_suffixes   = ["_" + "_".join(f.split("_")[1:])[:-4] for f in type_files]
    type_extensions = [f.split("_")[-1][-4:] for f in type_files]

def make_path(i, s):
    return dataset_dirname + "/" + i + s

begin = 0
end   = int(len(indices) * 0.9)
if is_validation:
    begin = end
    end   = len(indices)
elif is_test:
    begin = 0
    end   = len(indices)

for i in range(begin, end):
    for rs in random_suffixes:
        ii = indices[i]
        
        probe_image_path = dataset_path + "/" + ii + type_suffixes[0] + rs + path_suffixes[0] + type_extensions[0]
        probe_image      = Image.open(probe_image_path)

        # Ignore images that are too low res, or can't be loaded
        if probe_image.size[0] < 256 or probe_image.size[1] < 256:
            continue

        line = str(probe_image.size[0]) + " " + str(probe_image.size[1])
        if not is_test:
            reference_path = make_path(ii, reference_suffix) + reference_extension
            line          += " " + reference_path

        if not is_test and has_path_samples:
            flow_path  = make_path(ii, flow_suffix) + rs + flow_extension
            line      += " " + flow_path

        for pi in range(len(path_suffixes)):
            for ti in range(len(type_suffixes)):
                line += " " + make_path(ii, type_suffixes[ti]) + rs + path_suffixes[pi] + type_extensions[ti]

        print(line)
