# A detected socket (hole) in the sorting board, robot base frame.

string shape              # triangle | square | hexagon | circle
float32 shape_confidence  # 0..1
geometry_msgs/Point center  # centre of the hole at board-top height
float32 yaw               # vertex angle of the polygon (rad), modulo 2*pi/n
float32 width             # approx. diameter of the hole (m)
