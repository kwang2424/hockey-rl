extends Node2D
## Replays a hockey-rl trajectory JSON exported by `python -m hockey.export`.
##
## Nothing here simulates anything -- it draws absolute state, frame by frame.
## That is the whole point of the arrangement: training keeps its ~55k
## steps/sec in numpy, and this viewer can be as elaborate as you like without
## costing a single step of throughput.
##
## Controls:
##   Space          play / pause
##   Left / Right   step one frame (hold to scrub)
##   , / .          slower / faster
##   R              restart
##   O              open a different trajectory file
##   Esc            quit
##
## Batch use (no display needed, useful for CI or a quick preview):
##   godot --path viewer/godot -- --traj game.json --shot preview.png --frame 60

const DEFAULT_TRAJECTORY := "res://game.json"

var traj: Dictionary = {}
var frames: Array = []
var rink: Dictionary = {}
var teams: Array = []

var frame_idx: float = 0.0
var playing: bool = true
var speed: float = 1.0
var fps: float = 30.0
var status: String = ""
var _drew_once: bool = false
var _shot_path: String = ""
var _shot_frame: int = 60
var _shot_wait: int = 0

# World metres -> screen pixels.
var scale_px: float = 18.0
var origin: Vector2 = Vector2.ZERO

@onready var hud: Label = _make_hud()


func _parse_cli() -> Dictionary:
	## Reads args after the `--` separator, so they never collide with Godot's own.
	var out := {}
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		var a: String = args[i]
		if a.begins_with("--") and i + 1 < args.size():
			out[a.substr(2)] = args[i + 1]
			i += 2
		else:
			i += 1
	return out


func _ready() -> void:
	get_viewport().size_changed.connect(_recompute_transform)
	var cli := _parse_cli()
	if cli.has("traj"):
		_load_trajectory(str(cli["traj"]))
		if cli.has("shot"):
			_shot_path = str(cli["shot"])
			_shot_frame = int(cli.get("frame", "60"))
		return
	if not _load_trajectory(DEFAULT_TRAJECTORY):
		status = "No trajectory loaded. Press O to open a .json, or export one with:\n" \
			+ "    python -m hockey.export --a runs/v1/best.pt --b chase --out game.json"


func _make_hud() -> Label:
	var l := Label.new()
	l.position = Vector2(14, 10)
	l.add_theme_color_override("font_color", Color(0.90, 0.92, 0.96))
	l.add_theme_font_size_override("font_size", 15)
	add_child(l)
	return l


func _load_trajectory(path: String) -> bool:
	var text := ""
	if path.begins_with("res://"):
		if not ResourceLoader.exists(path) and not FileAccess.file_exists(path):
			return false
		var f := FileAccess.open(path, FileAccess.READ)
		if f == null:
			return false
		text = f.get_as_text()
	else:
		if not FileAccess.file_exists(path):
			status = "File not found: %s" % path
			return false
		var f2 := FileAccess.open(path, FileAccess.READ)
		if f2 == null:
			status = "Could not open: %s" % path
			return false
		text = f2.get_as_text()

	var parsed = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		status = "Not valid JSON: %s" % path
		return false
	if parsed.get("format", "") != "hockey-rl-trajectory":
		status = "Not a hockey-rl trajectory: %s" % path
		return false

	traj = parsed
	frames = traj.get("frames", [])
	rink = traj.get("rink", {})
	teams = traj.get("teams", [])
	fps = float(traj.get("fps", 30.0))
	frame_idx = 0.0
	status = ""
	_recompute_transform()
	print("[viewer] loaded %s: %d frames @ %.0f fps, %s vs %s, final %s" % [
		path, frames.size(), fps,
		teams[0].get("name", "A") if teams.size() > 0 else "A",
		teams[1].get("name", "B") if teams.size() > 1 else "B",
		str(traj.get("final_score", [])),
	])
	return frames.size() > 0


func _recompute_transform() -> void:
	if rink.is_empty():
		return
	var vp := get_viewport_rect().size
	var pad := 1.5
	var w: float = float(rink["length"]) + 2.0 * pad
	var h: float = float(rink["width"]) + 2.0 * pad + 2.2   # room for the HUD
	scale_px = min(vp.x / w, vp.y / h)
	origin = vp * 0.5 + Vector2(0, 12)


func _to_px(p: Vector2) -> Vector2:
	# World +y is up; screen +y is down.
	return origin + Vector2(p.x, -p.y) * scale_px


func _process(delta: float) -> void:
	if frames.is_empty():
		hud.text = status
		queue_redraw()
		return

	if _shot_path != "":
		# Batch screenshot mode: park on the requested frame. The capture
		# itself happens at the end of this function, once the HUD has been
		# filled in -- otherwise the saved PNG has a blank scoreboard.
		frame_idx = float(min(_shot_frame, frames.size() - 1))
		playing = false
	elif playing:
		frame_idx += delta * fps * speed
		if frame_idx >= frames.size():
			frame_idx = 0.0          # loop

	var f: Dictionary = frames[int(frame_idx) % frames.size()]
	var score: Array = f.get("score", [0, 0])
	var name_a: String = teams[0].get("name", "A") if teams.size() > 0 else "A"
	var name_b: String = teams[1].get("name", "B") if teams.size() > 1 else "B"
	var poss: int = int(f.get("possessor", -1))
	var poss_txt := "loose"
	if poss == 0:
		poss_txt = name_a
	elif poss == 1:
		poss_txt = name_b

	hud.text = "%s  %d - %d  %s        t=%.1fs   frame %d/%d   x%.2f   %s   puck: %s" % [
		name_a, int(score[0]), int(score[1]), name_b,
		float(f.get("t", 0.0)), int(frame_idx), frames.size(), speed,
		"playing" if playing else "PAUSED", poss_txt,
	]
	if f.has("event"):
		hud.text += "     << %s >>" % str(f["event"]).to_upper()
	queue_redraw()

	if _shot_path != "":
		# Let a couple of full frames render (HUD included) before grabbing the
		# viewport. Driven by a counter rather than `await`, because awaiting
		# inside _process starts a fresh coroutine on every single frame.
		_shot_wait += 1
		if _shot_wait >= 3:
			var img := get_viewport().get_texture().get_image()
			var err := img.save_png(_shot_path)
			print("[viewer] wrote %s (%s)" % [_shot_path, "ok" if err == OK else str(err)])
			get_tree().quit()


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed):
		return
	match event.keycode:
		KEY_SPACE:
			playing = not playing
		KEY_RIGHT:
			playing = false
			frame_idx = fposmod(frame_idx + 1.0, float(max(frames.size(), 1)))
		KEY_LEFT:
			playing = false
			frame_idx = fposmod(frame_idx - 1.0, float(max(frames.size(), 1)))
		KEY_COMMA:
			speed = max(0.1, speed * 0.5)
		KEY_PERIOD:
			speed = min(8.0, speed * 2.0)
		KEY_R:
			frame_idx = 0.0
			playing = true
		KEY_O:
			_open_dialog()
		KEY_ESCAPE:
			get_tree().quit()


func _open_dialog() -> void:
	var dlg := FileDialog.new()
	dlg.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	dlg.access = FileDialog.ACCESS_FILESYSTEM
	dlg.filters = PackedStringArray(["*.json ; trajectory"])
	dlg.size = Vector2i(720, 480)
	add_child(dlg)
	dlg.file_selected.connect(func(p): _load_trajectory(p); dlg.queue_free())
	dlg.canceled.connect(func(): dlg.queue_free())
	dlg.popup_centered()


# ---------------------------------------------------------------- drawing

func _rink_boundary_points() -> PackedVector2Array:
	## Rounded rectangle: the Minkowski sum of a rectangle and a disc, which is
	## exactly the shape the Python SDF defines, so the drawing cannot drift
	## from the collision geometry.
	var pts := PackedVector2Array()
	var r: float = float(rink["corner_radius"])
	var ix: float = float(rink["length"]) * 0.5 - r
	var iy: float = float(rink["width"]) * 0.5 - r
	var corners := [Vector2(ix, iy), Vector2(-ix, iy), Vector2(-ix, -iy), Vector2(ix, -iy)]
	var start := [0.0, PI * 0.5, PI, PI * 1.5]
	for c in range(4):
		for i in range(17):
			var a: float = start[c] + PI * 0.5 * (float(i) / 16.0)
			pts.append(_to_px(corners[c] + Vector2(cos(a), sin(a)) * r))
	return pts


func _draw() -> void:
	if rink.is_empty() or frames.is_empty():
		return
	if not _drew_once:
		_drew_once = true
		print("[viewer] first frame drawn OK (scale %.1f px/m)" % scale_px)

	var ice := Color(0.933, 0.957, 0.980)
	var red := Color(0.769, 0.157, 0.188)
	var blue := Color(0.157, 0.345, 0.745)
	var boards := Color(0.110, 0.133, 0.173)

	var outline := _rink_boundary_points()
	draw_colored_polygon(outline, ice)
	draw_polyline(outline + PackedVector2Array([outline[0]]), boards, 3.0, true)

	var hw: float = float(rink["width"]) * 0.5
	var glx: float = float(rink["goal_line_x"])
	var ghw: float = float(rink["goal_half_width"])
	var gd: float = float(rink["goal_depth"])
	var length: float = float(rink["length"])

	draw_line(_to_px(Vector2(0, -hw)), _to_px(Vector2(0, hw)), red, 0.30 * scale_px)
	for bx in [-length / 6.0, length / 6.0]:
		draw_line(_to_px(Vector2(bx, -hw)), _to_px(Vector2(bx, hw)), blue, 0.30 * scale_px)
	draw_arc(_to_px(Vector2.ZERO), 4.5 * scale_px, 0, TAU, 96, blue, 2.0, true)

	for sgn in [1.0, -1.0]:
		var gx: float = sgn * glx
		draw_line(_to_px(Vector2(gx, -hw)), _to_px(Vector2(gx, hw)), red, 2.0)
		draw_arc(_to_px(Vector2(gx - sgn * 0.05, 0)), 1.8 * scale_px, 0, TAU, 64,
			Color(0.588, 0.784, 0.941), 2.0, true)
		var net := PackedVector2Array([
			_to_px(Vector2(gx, ghw)), _to_px(Vector2(gx + sgn * gd, ghw)),
			_to_px(Vector2(gx + sgn * gd, -ghw)), _to_px(Vector2(gx, -ghw)),
		])
		draw_colored_polygon(net, Color(0.886, 0.886, 0.910))
		draw_polyline(net + PackedVector2Array([net[0]]), red, 2.0, true)
		for sy in [1.0, -1.0]:
			draw_circle(_to_px(Vector2(gx, sy * ghw)), 0.13 * scale_px, red)

	var f: Dictionary = frames[int(frame_idx) % frames.size()]
	var poss: int = int(f.get("possessor", -1))
	var sr: float = float(rink["skater_radius"])
	var blade: float = float(rink["blade_offset"])

	for a in range(2):
		var s: Array = f["skaters"][a]
		var pos := Vector2(float(s[0]), float(s[1]))
		var th: float = float(s[2])
		var col := red if a == 0 else blue
		if teams.size() > a and teams[a].has("colour"):
			col = Color.from_string(str(teams[a]["colour"]), col)

		if poss == a:
			draw_arc(_to_px(pos), (sr + 0.22) * scale_px, 0, TAU, 48,
				Color(1.0, 0.839, 0.314), 3.0, true)

		var tip := pos + Vector2(cos(th), sin(th)) * blade
		draw_line(_to_px(pos), _to_px(tip), Color(0.235, 0.173, 0.125), 0.13 * scale_px)
		draw_circle(_to_px(pos), sr * scale_px, col)
		draw_arc(_to_px(pos), sr * scale_px, 0, TAU, 32, Color(0.08, 0.08, 0.09), 1.5, true)
		# Nose marker, so heading is readable at a glance.
		draw_circle(_to_px(pos + Vector2(cos(th), sin(th)) * sr * 0.55), sr * 0.30 * scale_px,
			Color(0.98, 0.98, 0.99))
		draw_circle(_to_px(tip), 0.15 * scale_px, Color(0.235, 0.173, 0.125))

	var puck := Vector2(float(f["puck"][0]), float(f["puck"][1]))
	draw_circle(_to_px(puck), max(float(rink["puck_radius"]) * 3.2, 0.16) * scale_px,
		Color(0.07, 0.07, 0.08))
