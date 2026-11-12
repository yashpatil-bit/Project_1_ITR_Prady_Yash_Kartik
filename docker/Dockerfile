# ROS 2 Humble image to build and run the stack (mock mode works anywhere;
# for the real robot/camera run natively, or with --privileged -v /dev:/dev).
#   docker build -t peg_sorting -f docker/Dockerfile .
#   docker run -it --rm peg_sorting bash -c "./launch/run_autonomy.sh --mock --no-confirm --no-window"
FROM ros:humble-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-pytest ros-humble-control-msgs ros-humble-dynamixel-sdk python3-serial \
      libgl1 libglib2.0-0 libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /root/peg_ws/src/peg_sorting_repo
COPY . .
WORKDIR /root/peg_ws
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install
RUN echo "source /opt/ros/humble/setup.bash && source /root/peg_ws/install/setup.bash" >> /root/.bashrc
WORKDIR /root/peg_ws/src/peg_sorting_repo
ENV PEG_WS=/root/peg_ws
