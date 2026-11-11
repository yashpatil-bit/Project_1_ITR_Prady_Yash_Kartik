# A detected peg. All coordinates are in the robot base frame
# (origin on the table directly below joint1, x forward, y left, z up), metres / radians.

string color              # yellow | blue | orange | green  (keys of colors.yaml)
string shape              # triangle | square | hexagon | circle | unknown
string shape_source       # geometry | yolo | color_prior
float32 shape_confidence  # 0..1

geometry_msgs/Point top_center   # centre of the peg's top face
float32 height            # peg height above the table (m)
float32 yaw               # vertex angle of the polygon (rad), meaningful modulo 2*pi/n
float32 width             # approx. diameter of the footprint (m)
int32 pixel_area          # size of the colour blob in the image
