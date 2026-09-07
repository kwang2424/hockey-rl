# Engine viewers

The sim is already decoupled from rendering, so "visualise it in Godot/Unity"
does **not** mean porting the physics. It means exporting state and replaying
it:

```
python -m hockey.export --a runs/v1/best.pt --b chase --out game.json
```

This is a much better arrangement than simulating inside the engine. Training
keeps its ~55k steps/sec in numpy, and the viewer can be as elaborate as you
like without costing a single step of throughput. It also means the viewer
never has to stay in sync with the physics — only with the schema, which is
pinned by `tests/test_export.py`.

## Godot (included, working)

```bash
python -m hockey.export --a runs/v1/best.pt --b chase --out viewer/godot/game.json
godot --path viewer/godot          # or open the folder in Godot 4.2+ and press F5
```

| key | action |
|---|---|
| `Space` | play / pause |
| `←` `→` | step one frame |
| `,` `.` | slower / faster |
| `R` | restart |
| `O` | open a different trajectory |
| `Esc` | quit |

It draws the rink from the `rink` block in the JSON rather than from hardcoded
constants, so if you change the rink in `hockey/config.py` the viewer follows
automatically. The rounded boards are reconstructed as the Minkowski sum of a
rectangle and a disc — exactly the shape the Python SDF defines — so the
drawing cannot drift from the collision geometry.

There is also a batch mode that needs no display, useful for CI or a quick
preview:

```bash
godot --path viewer/godot -- --traj game.json --shot preview.png --frame 150
```

Verified on Godot 4.2.2 (headless load + draw, and a screenshot under xvfb).

## Unity

There is no Unity project here — I can't run Unity in this environment, so
rather than ship something untested, here is the schema and the shape of the
script. Reading the JSON is genuinely all there is to it.

```csharp
using UnityEngine;

[System.Serializable] public class Rink {
    public float length, width, corner_radius, goal_line_x,
                 goal_half_width, goal_depth, skater_radius,
                 puck_radius, blade_offset;
}
[System.Serializable] public class Frame {
    public float t;
    public float[][] skaters;   // [[x, y, theta], [x, y, theta]]
    public float[] puck;        // [x, y]
    public int possessor;       // -1 loose, else skater index
    public int[] score;
    public string @event;       // null on ordinary frames
}
[System.Serializable] public class Trajectory {
    public string format; public int version; public float fps;
    public Rink rink; public Frame[] frames; public int[] final_score;
}

public class Replay : MonoBehaviour {
    public TextAsset trajectoryJson;
    public Transform skaterA, skaterB, puck;
    Trajectory traj; float t;

    void Start() {
        // Unity's JsonUtility does not handle jagged arrays; use Newtonsoft
        // Json.NET (com.unity.nuget.newtonsoft-json) or export a flattened
        // variant if you would rather avoid the dependency.
        traj = Newtonsoft.Json.JsonConvert.DeserializeObject<Trajectory>(trajectoryJson.text);
    }

    void Update() {
        t += Time.deltaTime * traj.fps;
        var f = traj.frames[Mathf.FloorToInt(t) % traj.frames.Length];
        Place(skaterA, f.skaters[0]);
        Place(skaterB, f.skaters[1]);
        puck.position = new Vector3(f.puck[0], 0f, f.puck[1]);
    }

    // World is x/y with +y up; Unity is x/z with +y up, so y -> z.
    void Place(Transform tr, float[] s) {
        tr.position = new Vector3(s[0], 0f, s[1]);
        tr.rotation = Quaternion.Euler(0f, -s[2] * Mathf.Rad2Deg, 0f);
    }
}
```

Two things worth knowing before you start:

- **Axis convention.** The sim is 2D with `+y` up the screen. Unity's ground
  plane is x/z, so map `y -> z` and negate the heading when converting to a
  Y-axis rotation (as above). Godot is 2D with `+y` *down*, which is why
  `Main.gd` flips y in `_to_px`.
- **Units are already metres**, so a 1:1 scale works directly. The rink is
  60m x 26m, skaters are 0.45m radius, the puck 0.038m.

## Schema

See the docstring at the top of `hockey/export.py` for the authoritative
version. Frames are **absolute state, not deltas**, so a viewer can scrub to
any index without replaying from the start. `tests/test_export.py` pins the
contract — including that `score` is monotonic and consistent with the `event`
markers, and that a claimed `possessor` really is within stick reach of the
puck.
