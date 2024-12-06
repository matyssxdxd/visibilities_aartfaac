import struct

import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy.ma.extras import average


class Header:
    def __init__(self):
        self.magic = None
        self.nr_receivers = None
        self.nr_polarizations = None
        self.correlation_mode = None
        self.start_time = None
        self.end_time = None
        self.weights = None
        self.nr_samples_per_integration = None
        self.nr_channels = None
        self.pad0 = None
        self.first_channel_frequency = None
        self.channel_bandwidth = None
        self.pad1 = None


import struct
import numpy as np


def read(file_path):
    headers = []
    visibilities = []
    read_count = 0

    with open(file_path, 'rb') as file:
        while True:
            try:
                # Read header
                header = Header()
                header.magic = struct.unpack('I', file.read(4))[0]
                header.nr_receivers = struct.unpack('H', file.read(2))[0]
                header.nr_polarizations = struct.unpack('B', file.read(1))[0]
                header.correlation_mode = struct.unpack('B', file.read(1))[0]
                header.start_time = struct.unpack('d', file.read(8))[0]
                header.end_time = struct.unpack('d', file.read(8))[0]
                header.weights = struct.unpack('I' * 300, file.read(4 * 300))
                header.nr_samples_per_integration = struct.unpack('I', file.read(4))[0]
                header.nr_channels = struct.unpack('H', file.read(2))[0]
                header.pad0 = file.read(2)
                header.first_channel_frequency = struct.unpack('d', file.read(8))[0]
                header.channel_bandwidth = struct.unpack('d', file.read(8))[0]
                header.pad1 = file.read(288)

                vis_dtype = np.complex64
                nr_baselines = header.nr_receivers + int(header.nr_receivers * (header.nr_receivers - 1) / 2)
                vis_shape = (nr_baselines, header.nr_channels, header.nr_polarizations)
                vis_zeros = np.zeros(vis_shape, vis_dtype)
                # Read visibilities
                vis = file.read(vis_zeros.size * vis_zeros.itemsize)

                if len(vis) < vis_zeros.size * vis_zeros.itemsize:
                    break

                vis = np.frombuffer(vis, dtype=vis_dtype).reshape(vis_shape)
                headers.append(header)
                visibilities.append(vis)
                read_count += 1
            except struct.error:
                break
    return headers, visibilities


def print_header(header):
    print("Printing header...")
    print(f"magic: {header.magic}")
    print(f"nrReceivers: {header.nr_receivers}")
    print(f"nrPolarizations: {header.nr_polarizations}")
    print(f"correlationMode: {header.correlation_mode}")
    print(f"startTime: {header.start_time}")
    print(f"endTime: {header.end_time}")
    print(f"weights: {header.weights}")
    print(f"nrSamplesPerIntegration: {header.nr_samples_per_integration}")
    print(f"nrChannels: {header.nr_channels}")
    print(f"pad0: {header.pad0}")
    print(f"firstChannelFrequency: {header.first_channel_frequency}")
    print(f"channelBandwidth: {header.channel_bandwidth}")
    print(f"pad1: {header.pad1}")

if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        print(sys.argv[i])
        header, visibilities = read(sys.argv[i])
        results = [[[[] for _ in range(255)] for _ in range(4)] for _ in range(3)]

        for vis in range(len(visibilities)):
            for baseline in range(len(visibilities[vis])):
                for channel in range(len(visibilities[vis][baseline])):
                    for polarization in range(len(visibilities[vis][baseline][channel])):
                        results[baseline][polarization][channel].append(visibilities[vis][baseline][channel][polarization].real)

        for baseline in range(len(results)):
            if baseline == 0 or baseline == 2:
                for polarization in range(len(results[baseline])):
                    if polarization == 0 or polarization == 3:
                        for channel in range(len(results[baseline][polarization])):
                            temp = average(results[baseline][polarization][channel])
                            results[baseline][polarization][channel] = temp

        for baseline in range(len(results)):
            if baseline == 0 or baseline == 2:
                for polarization in range(len(results[baseline])):
                    if polarization == 0 or polarization == 3:
                        file_name = str(sys.argv[i].split(".")[0]) + "_"
                        x_points = np.arange(len(results[baseline][polarization]))
                        y_points = results[baseline][polarization]

                        plt.figure()
                        plt.plot(x_points, y_points)

                        if baseline == 0:
                            file_name += "ib"
                        elif baseline == 2:
                            file_name += "ir"

                        if polarization == 0:
                            file_name += "_rr"
                        elif polarization == 3:
                            file_name += "_ll"

                        file_name += ".png"

                        plt.savefig(file_name)
                        plt.close()