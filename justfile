set shell := ["bash", "-cu"]

archive := "/home/kasm-user/Downloads/Color_000200_001000_ccw_frames.tar.gz"
frames := "data/Color_000200_001000_ccw_frames"
out := "eval_results/demo/color_000200_001000_512_chunk16"
model := "mda_mog_sky_l2"
view_result := "eval_results/demo/color_000200_001000_512_balanced_chunk16/mda_mog_sky_l2"
view_rrd := "eval_results/demo/color_000200_001000_512_balanced_chunk16/mda_mog_sky_l2/mda_result_p95_stride2.rrd"

default:
    just --list

sync:
    uv sync

weights:
    uv run hf download sy000/MDA --local-dir checkpoints/MDA

extract archive=archive:
    mkdir -p data
    tar -xzf "{{archive}}" -C data
    find "{{frames}}" -maxdepth 1 -type f -name '*.jpg' | wc -l

inspect dir=frames:
    find "{{dir}}" -maxdepth 1 -type f | sort | sed -n '1,5p'
    find "{{dir}}" -maxdepth 1 -type f | sort | tail -5
    find "{{dir}}" -maxdepth 1 -type f -name '*.jpg' | wc -l

smoke dir=frames out="eval_results/demo/color_smoke_512" cooldown="0":
    uv run python src/testing/run_inference_video.py \
        --model_name "{{model}}" \
        --img_path "{{dir}}" \
        --output_dir "{{out}}" \
        --size 512 \
        --max_chunk 1 \
        --gpu_cooldown_sec "{{cooldown}}" \
        --render_sideview 0 \
        --image_glob "*.jpg" \
        --image_regex '^00020[0-3]\.jpg$' \
        --sort_mode natural \
        --scene_id color_smoke

infer dir=frames out=out size="512" max_chunk="16" sideview="0" glob="*.jpg" regex=".*" sort="natural" cooldown="0":
    uv run python -u src/testing/run_inference_video.py \
        --model_name "{{model}}" \
        --img_path "{{dir}}" \
        --output_dir "{{out}}" \
        --size "{{size}}" \
        --max_chunk "{{max_chunk}}" \
        --gpu_cooldown_sec "{{cooldown}}" \
        --render_sideview "{{sideview}}" \
        --image_glob "{{glob}}" \
        --image_regex "{{regex}}" \
        --sort_mode "{{sort}}" \
        --scene_id "$(basename "{{dir}}")"

background dir=frames out=out size="512" max_chunk="8" sideview="0" glob="*.jpg" regex=".*" sort="natural" cooldown="0.5":
    mkdir -p eval_results/logs
    log="eval_results/logs/$(basename "{{out}}").log"; \
    pidfile="eval_results/logs/$(basename "{{out}}").pid"; \
    setsid bash -c 'echo "started $(date)"; uv run python -u src/testing/run_inference_video.py \
        --model_name "{{model}}" \
        --img_path "{{dir}}" \
        --output_dir "{{out}}" \
        --size "{{size}}" \
        --max_chunk "{{max_chunk}}" \
        --gpu_cooldown_sec "{{cooldown}}" \
        --render_sideview "{{sideview}}" \
        --image_glob "{{glob}}" \
        --image_regex "{{regex}}" \
        --sort_mode "{{sort}}" \
        --scene_id "$(basename "{{dir}}")"; \
        status=$?; echo "finished $(date) status=$status"; exit $status' \
        > "$log" 2>&1 < /dev/null & \
    echo $! > "$pidfile"; \
    echo "started pid=$(cat "$pidfile")"; \
    echo "log=$log"; \
    echo "progress={{out}}/{{model}}/progress.json"

status out=out:
    @pidfile="eval_results/logs/$(basename "{{out}}").pid"; \
    if [[ -f "$pidfile" ]]; then \
        pid="$(cat "$pidfile")"; \
        ps -p "$pid" -o pid=,stat=,etime=,cmd= || true; \
        pgrep -P "$pid" -af || true; \
    else \
        echo "pidfile not found: $pidfile"; \
    fi
    @progress="{{out}}/{{model}}/progress.json"; \
    if [[ -f "$progress" ]]; then \
        uv run python -m json.tool "$progress"; \
    else \
        echo "progress not found: $progress"; \
    fi
    @OUT="{{out}}/{{model}}"; \
    printf 'png '; find "$OUT" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l; \
    printf 'npy '; find "$OUT" -maxdepth 1 -name 'frame_*.npy' 2>/dev/null | wc -l; \
    printf 'raw '; find "$OUT/raw" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l

tail-log out=out lines="80":
    @log="eval_results/logs/$(basename "{{out}}").log"; \
    tail -n "{{lines}}" "$log"

stop out=out:
    @pidfile="eval_results/logs/$(basename "{{out}}").pid"; \
    if [[ -f "$pidfile" ]]; then \
        pid="$(cat "$pidfile")"; \
        pkill -TERM -P "$pid" 2>/dev/null || true; \
        kill "$pid" 2>/dev/null || true; \
        echo "sent TERM to pid=$pid and children"; \
    else \
        echo "pidfile not found: $pidfile"; \
    fi

viewer result=view_result host="0.0.0.0" port="7861":
    uv run python src/testing/result_viewer.py \
        --result_dir "{{result}}" \
        --host "{{host}}" \
        --port "{{port}}"

viewer-bg result=view_result host="0.0.0.0" port="7861":
    mkdir -p eval_results/logs
    log="eval_results/logs/result_viewer_{{port}}.log"; \
    pidfile="eval_results/logs/result_viewer_{{port}}.pid"; \
    setsid bash -c 'uv run python src/testing/result_viewer.py \
        --result_dir "{{result}}" \
        --host "{{host}}" \
        --port "{{port}}"' \
        > "$log" 2>&1 < /dev/null & \
    echo $! > "$pidfile"; \
    echo "started pid=$(cat "$pidfile")"; \
    echo "log=$log"; \
    echo "url=http://127.0.0.1:{{port}}"

viewer-status port="7861":
    @pidfile="eval_results/logs/result_viewer_{{port}}.pid"; \
    if [[ -f "$pidfile" ]]; then \
        pid="$(cat "$pidfile")"; \
        ps -p "$pid" -o pid=,stat=,etime=,cmd= || true; \
        pgrep -P "$pid" -af || true; \
    else \
        echo "pidfile not found: $pidfile"; \
    fi
    @log="eval_results/logs/result_viewer_{{port}}.log"; \
    if [[ -f "$log" ]]; then tail -n 30 "$log"; fi

viewer-stop port="7861":
    @pidfile="eval_results/logs/result_viewer_{{port}}.pid"; \
    if [[ -f "$pidfile" ]]; then \
        pid="$(cat "$pidfile")"; \
        pkill -TERM -P "$pid" 2>/dev/null || true; \
        kill "$pid" 2>/dev/null || true; \
        echo "sent TERM to pid=$pid and children"; \
    else \
        echo "pidfile not found: $pidfile"; \
    fi

viewer-rrd result=view_result rrd="" max_points="12000" frame_stride="2" depth_percentile="95":
    @out="{{rrd}}"; \
    if [[ -z "$out" ]]; then out="{{result}}/mda_result_p95_stride2.rrd"; fi; \
    uv run --extra viz python src/testing/result_to_rerun.py \
        --result_dir "{{result}}" \
        --rrd "$out" \
        --max_points "{{max_points}}" \
        --frame_stride "{{frame_stride}}" \
        --depth_percentile "{{depth_percentile}}"

rerun-open rrd=view_rrd:
    uv run --extra viz rerun "{{rrd}}"

rerun-serve rrd=view_rrd web_port="9090" grpc_port="9876" bind="0.0.0.0":
    @echo "web viewer: http://127.0.0.1:{{web_port}}"
    @echo "remote viewer URL usually needs this host IP instead of 127.0.0.1:"
    @echo "http://<host-ip>:{{web_port}}/?url=rerun%2Bhttp%3A%2F%2F<host-ip>%3A{{grpc_port}}%2Fproxy"
    uv run --extra viz rerun --serve-web --bind "{{bind}}" --web-viewer-port "{{web_port}}" --port "{{grpc_port}}" "{{rrd}}"
