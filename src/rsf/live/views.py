"""Moving the cameras from the viewer (shared: one picture per view, everyone sees the
camera where the last viewer put it; no token: it does not touch the simulation).

==================================================  ==================================================
``GET /api/v1/views``                               every view's pose: eye, target, yaw, pitch, fov_deg
``POST /api/v1/views/<name>/pose``                  ``{"eye": [x, y, z], "yaw": deg, "pitch": deg,
                                                    "fov_deg": deg}`` (or ``"target"`` for yaw/pitch;
                                                    missing parts stay)
``POST /api/v1/views/<name>/reset``                 back to the view as the run started it
==================================================  ==================================================

Yaw turns about the world z from +x (counter-clockwise seen from above), pitch is up
from the horizontal (within +-89.9 deg). The capture applies a move on the simulation
thread before the next frame and draws at its interactive rate meanwhile; the status
(``views``) carries the poses, so every viewer follows.
"""
from __future__ import annotations


def register(server, capture) -> None:
    server.features["camera"] = {"views": [v.name for v in capture.views]}

    @server.route("GET", r"/api/v1/views")
    def views(req, m):
        req.send_json({v.name: capture.view_pose(v.name) for v in capture.views})

    @server.route("POST", r"/api/v1/views/(?P<name>[^/]+)/pose")
    def pose(req, m):
        try:
            b = req.json_body()
            req.send_json(capture.request_view(m["name"], eye=b.get("eye"), target=b.get("target"), yaw=b.get("yaw"),
                                               pitch=b.get("pitch"), fov_deg=b.get("fov_deg")))
        except KeyError:
            req.send_json({"error": f"no view {m['name']!r}"}, 404)
        except (TypeError, ValueError) as e:
            req.send_json({"error": str(e)}, 400)

    @server.route("POST", r"/api/v1/views/(?P<name>[^/]+)/reset")
    def reset(req, m):
        try:
            req.send_json(capture.reset_view(m["name"]))
        except KeyError:
            req.send_json({"error": f"no view {m['name']!r}"}, 404)
