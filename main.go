package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

//go:embed static/index.html
var staticFiles embed.FS

const (
	maxUserSessions = 4
	maxSessionBytes = 128 * 1024
	maxSessionEvents = 160
	maxDownloadSessionBytes = 32 * 1024
	maxDownloadSessionEvents = 48
	maxSessionSubscribers = 16
	maxFileEntries = 4000
	maxDownloadDirEntries = 2000
	requestBodyLimit = 1 << 20
	maxTerminalInputBytes = 4096
	maxDownloadQueue = 64

	defaultAppRoot = "/home/simonsyoyo/Comfyui"
	defaultDataRoot = "/home/simonsyoyo/Comfyui/comfy_container_data"
	defaultModelsRoot = "/home/simonsyoyo/Comfyui/ComfyUI/models"
	defaultContainerRoot = "/opt/ComfyUI"
	defaultClashVergeConfig = "~/.config/io.github.clash-verge-rev.clash-verge-rev/verge.yaml"

	tiocgptn = 0x80045430
	tiocsptlck = 0x40045431
)

type Config struct {
	ConfigPath        string   `json:"configPath,omitempty"`
	BindHost          string   `json:"bindHost"`
	Port              int      `json:"port"`
	ContainerName     string   `json:"containerName"`
	VllmContainerName string   `json:"vllmContainerName"`
	AppRoot           string   `json:"appRoot"`
	DataRoot          string   `json:"dataRoot"`
	ContainerDataRoot string   `json:"containerDataRoot"`
	ModelsPath        string   `json:"modelsPath"`
	OutputPath        string   `json:"outputPath"`
	CustomNodesPath   string   `json:"customNodesPath"`
	InputPath         string   `json:"inputPath"`
	UserPath          string   `json:"userPath"`
	DownloadDirs      []string `json:"downloadDirs"`

	ClashControllerURL            string `json:"clashControllerUrl"`
	ClashSecret                   string `json:"clashSecret"`
	ClashVergeConfigPath          string `json:"clashVergeConfigPath"`
	ClashSystemProxyOnCommand     string `json:"clashSystemProxyOnCommand"`
	ClashSystemProxyOffCommand    string `json:"clashSystemProxyOffCommand"`
	ClashRuleModeCommand          string `json:"clashRuleModeCommand"`
	ClashGlobalModeCommand        string `json:"clashGlobalModeCommand"`
	ClashDirectModeCommand        string `json:"clashDirectModeCommand"`
	ClashStartCommand             string `json:"clashStartCommand"`
	ClashStopCommand              string `json:"clashStopCommand"`
	HTTPProxy                     string `json:"httpProxy"`
	SocksProxy                    string `json:"socksProxy"`
	ControlButtons                []string `json:"controlButtons"`
	VllmServiceStartCommand       string   `json:"vllmServiceStartCommand"`
	VllmServiceStopCommand        string   `json:"vllmServiceStopCommand"`
}

func defaultConfig() Config {
	return Config{
		BindHost:          "127.0.0.1",
		Port:              8848,
		ContainerName:     "comfyui",
		VllmContainerName: "vllm-qwen35b",
		AppRoot:           defaultAppRoot,
		DataRoot:          defaultDataRoot,
		ContainerDataRoot: defaultContainerRoot,
		ModelsPath:        defaultModelsRoot,
		OutputPath:        defaultDataRoot + "/output",
		CustomNodesPath:   defaultDataRoot + "/custom_nodes",
		InputPath:         defaultDataRoot + "/input",
		UserPath:          defaultDataRoot + "/user",
		DownloadDirs: []string{
			defaultModelsRoot + "/checkpoints",
			defaultModelsRoot + "/loras",
			defaultModelsRoot + "/vae",
			defaultDataRoot + "/custom_nodes",
			defaultDataRoot + "/input",
			defaultDataRoot + "/output",
			defaultDataRoot + "/user",
		},
		ClashControllerURL:    "http://127.0.0.1:9090",
		ClashVergeConfigPath: defaultClashVergeConfig,
		ClashStartCommand:     "clash-verge",
		ClashStopCommand:      "pkill -TERM -f 'clash-verge|Clash Verge'",
		HTTPProxy:             "http://127.0.0.1:7890",
		SocksProxy:            "socks5://127.0.0.1:7891",
		VllmServiceStartCommand: "docker exec vllm-qwen35b bash -c 'vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --trust-remote-code --kv-cache-dtype fp8 --attention-backend flashinfer --moe-backend marlin --gpu-memory-utilization 0.4 --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling --enable-prefix-caching --speculative-config '\"'\"'{\"method\":\"mtp\",\"num_speculative_tokens\":1,\"moe_backend\":\"triton\"}'\"'\"' --load-format fastsafetensors --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice'",
		VllmServiceStopCommand:  "docker exec vllm-qwen35b pkill -f 'vllm serve'",
		ControlButtons:        []string{"Ctrl+C 中断", "Ctrl+D 结束输入", "Ctrl+Z 挂起", "Ctrl+L 清屏", "Tab 补全", "Esc 取消"},
	}
}

type Server struct {
	cfgMu sync.RWMutex
	cfg Config
	httpServer *http.Server
	serviceOnce sync.Once

	sessions *SessionHub
	downloads chan DownloadTask

	dlMu sync.RWMutex
	downloadProxy bool

	clashMu sync.RWMutex
	clashMode string
	clashAPI bool
	clashAPIChecked bool
	clashRunning bool
	clashRunningChecked bool

	vllmMu sync.RWMutex
	vllmServiceStarting bool
}

type Event struct {
	Seq uint64 `json:"seq"`
	Data string `json:"data"`
	At time.Time `json:"at"`
}

type Session struct {
	mu sync.Mutex
	ID string `json:"id"`
	Title string `json:"title"`
	CWD string `json:"cwd"`
	Kind string `json:"kind"`
	Fixed bool `json:"fixed"`
	Closable bool `json:"closable"`
	Running bool `json:"running"`

	cmd *exec.Cmd
	stdin io.WriteCloser
	pty *os.File
	done chan error
	trackCWD bool

	seq uint64
	byteCount int
	events []Event
	subs map[chan Event]struct{}
	MaxBytes int
	MaxEvents int
}

type SessionHub struct {
	mu sync.Mutex
	sessions map[string]*Session
	nextID int
}

type DownloadTask struct {
	ID int64 `json:"id"`
	Method string `json:"method"`
	Dir string `json:"dir"`
	URL string `json:"url"`
	UseProxy bool `json:"useProxy"`
}

type FileEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
	Type string `json:"type"`
	Size int64 `json:"size"`
	ModTime time.Time `json:"modTime"`
}

type ManagedRoot struct {
	Name string
	Host string
	Container string
}

type DownloadDirEntry struct {
	Path string `json:"path"`
	Label string `json:"label"`
}

func main() {
	cfg := defaultConfig()
	if err := loadConfig("config.json", &cfg); err != nil {
		log.Printf("using default config: %v", err)
	}

	normalizeConfig(&cfg)
	cfg.BindHost = loopbackBindHost(cfg.BindHost)

	s := &Server{
		cfg: cfg,
		sessions: NewSessionHub(),
		downloads: make(chan DownloadTask, maxDownloadQueue),
		clashMode: "rule",
	}
	s.initFixedSessions()
	go s.downloadWorker()

	mux := http.NewServeMux()
	s.routes(mux)

	addr := net.JoinHostPort(cfg.BindHost, strconv.Itoa(cfg.Port))
	server := &http.Server{
		Addr:              addr,
		Handler:           localOnly(withSecurityHeaders(recoverMiddleware(mux))),
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 16,
	}
	s.httpServer = server

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		s.shutdownSessions()
		_ = server.Shutdown(shutdownCtx)
	}()

	log.Printf("ComfyUI manager listening on http://%s", addr)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func loadConfig(path string, cfg *Config) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, cfg)
}

func saveConfig(path string, cfg Config) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}



func normalizeConfig(cfg *Config) {
	cfg.ConfigPath = ""
	cfg.AppRoot = defaultAppRoot
	cfg.DataRoot = defaultDataRoot
	cfg.ModelsPath = defaultModelsRoot
	if strings.TrimSpace(cfg.OutputPath) == "" {
		cfg.OutputPath = defaultDataRoot + "/output"
	}
	if strings.TrimSpace(cfg.CustomNodesPath) == "" {
		cfg.CustomNodesPath = defaultDataRoot + "/custom_nodes"
	}
	if strings.TrimSpace(cfg.InputPath) == "" {
		cfg.InputPath = defaultDataRoot + "/input"
	}
	if strings.TrimSpace(cfg.UserPath) == "" {
		cfg.UserPath = defaultDataRoot + "/user"
	}
	cfg.OutputPath = strings.TrimSpace(cfg.OutputPath)
	cfg.CustomNodesPath = strings.TrimSpace(cfg.CustomNodesPath)
	cfg.InputPath = strings.TrimSpace(cfg.InputPath)
	cfg.UserPath = strings.TrimSpace(cfg.UserPath)
	if strings.TrimSpace(cfg.BindHost) == "" {
		cfg.BindHost = "127.0.0.1"
	}
	if cfg.Port == 0 {
		cfg.Port = 8848
	}
	if strings.TrimSpace(cfg.ContainerName) == "" {
		cfg.ContainerName = "comfyui"
	}
	if strings.TrimSpace(cfg.VllmContainerName) == "" {
		cfg.VllmContainerName = "vllm-qwen35b"
	}
	if strings.TrimSpace(cfg.ContainerDataRoot) == "" {
		cfg.ContainerDataRoot = defaultContainerRoot
	}
	if strings.TrimSpace(cfg.ClashControllerURL) == "" {
		cfg.ClashControllerURL = "http://127.0.0.1:9090"
	}
	if strings.TrimSpace(cfg.ClashVergeConfigPath) == "" {
		cfg.ClashVergeConfigPath = defaultClashVergeConfig
	}
	cfg.ClashSystemProxyOnCommand = strings.TrimSpace(cfg.ClashSystemProxyOnCommand)
	cfg.ClashSystemProxyOffCommand = strings.TrimSpace(cfg.ClashSystemProxyOffCommand)
	cfg.ClashRuleModeCommand = strings.TrimSpace(cfg.ClashRuleModeCommand)
	cfg.ClashGlobalModeCommand = strings.TrimSpace(cfg.ClashGlobalModeCommand)
	cfg.ClashDirectModeCommand = strings.TrimSpace(cfg.ClashDirectModeCommand)
	if strings.TrimSpace(cfg.ClashStartCommand) == "" {
		cfg.ClashStartCommand = "clash-verge"
	}
	if strings.TrimSpace(cfg.ClashStopCommand) == "" {
		cfg.ClashStopCommand = "pkill -TERM -f 'clash-verge|Clash Verge'"
	}
	if strings.TrimSpace(cfg.HTTPProxy) == "" {
		cfg.HTTPProxy = "http://127.0.0.1:7890"
	}
	if strings.TrimSpace(cfg.SocksProxy) == "" {
		cfg.SocksProxy = "socks5://127.0.0.1:7891"
	}
	cfg.ControlButtons = cleanControlButtons(cfg.ControlButtons)
	if strings.TrimSpace(cfg.VllmServiceStartCommand) == "" {
		cfg.VllmServiceStartCommand = defaultConfig().VllmServiceStartCommand
	}
	if strings.TrimSpace(cfg.VllmServiceStopCommand) == "" {
		cfg.VllmServiceStopCommand = defaultConfig().VllmServiceStopCommand
	}
	cfg.DownloadDirs = filterManagedDownloadDirs(cfg.DownloadDirs)
	if len(cfg.DownloadDirs) == 0 {
		cfg.DownloadDirs = defaultConfig().DownloadDirs
	}
}

func cleanControlButtons(buttons []string) []string {
	out := make([]string, 0, min(len(buttons), 12))
	for _, item := range buttons {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		out = append(out, item)
		if len(out) >= 12 {
			break
		}
	}
	if len(out) == 0 {
		return []string{"Ctrl+C 中断", "Ctrl+D 结束输入", "Ctrl+Z 挂起", "Ctrl+L 清屏", "Tab 补全", "Esc 取消"}
	}
	return out
}

func NewSessionHub() *SessionHub {
	return &SessionHub{sessions: make(map[string]*Session)}
}

func NewSession(id, title, cwd, kind string, fixed, closable bool) *Session {
	return &Session{
		ID: id,
		Title: title,
		CWD: cwd,
		Kind: kind,
		Fixed: fixed,
		Closable: closable,
		Running: false,
		subs: make(map[chan Event]struct{}),
		MaxBytes: maxSessionBytes,
		MaxEvents: maxSessionEvents,
	}
}

func (h *SessionHub) Add(s *Session) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.sessions[s.ID] = s
}

func (h *SessionHub) Get(id string) (*Session, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	s, ok := h.sessions[id]
	return s, ok
}

func (h *SessionHub) Remove(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.sessions, id)
}

func (h *SessionHub) List() []*Session {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]*Session, 0, len(h.sessions))
	for _, s := range h.sessions {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Fixed != out[j].Fixed {
			return out[i].Fixed
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func (h *SessionHub) UserCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	n := 0
	for _, s := range h.sessions {
		if !s.Fixed {
			n++
		}
	}
	return n
}

func (h *SessionHub) NextUserID(prefix string) string {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.nextID++
	return fmt.Sprintf("%s-%d", prefix, h.nextID)
}

func (s *Session) Append(data string) {
	if data == "" {
		return
	}
	if len(data) > 8192 {
		data = safeTruncate(data, 8192) + "\n[output truncated]"
	}
	ev := Event{Data: data, At: time.Now()}

	s.mu.Lock()
	s.seq++
	ev.Seq = s.seq
	s.events = append(s.events, ev)
	s.byteCount += len(ev.Data)
	maxBytes := s.MaxBytes
	if maxBytes <= 0 {
		maxBytes = maxSessionBytes
	}
	maxEvents := s.MaxEvents
	if maxEvents <= 0 {
		maxEvents = maxSessionEvents
	}
	for len(s.events) > 0 && (s.byteCount > maxBytes || len(s.events) > maxEvents) {
		s.byteCount -= len(s.events[0].Data)
		s.events = s.events[1:]
	}
	// Compact backing array if capacity grew too large
	if cap(s.events) > 2*len(s.events) && len(s.events) > 0 {
		trimmed := make([]Event, len(s.events))
		copy(trimmed, s.events)
		s.events = trimmed
	}
	for ch := range s.subs {
		select {
		case ch <- ev:
		default:
		}
	}
	s.mu.Unlock()
}

func (s *Session) Clear() {
	ev := Event{Data: "\x1b[2J\x1b[H", At: time.Now()}

	s.mu.Lock()
	s.seq++
	ev.Seq = s.seq
	s.events = []Event{ev}
	s.byteCount = len(ev.Data)
	for ch := range s.subs {
		select {
		case ch <- ev:
		default:
		}
	}
	s.mu.Unlock()
}

func (s *Session) Snapshot(after uint64) []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Event, 0, len(s.events))
	for _, ev := range s.events {
		if ev.Seq > after {
			out = append(out, ev)
		}
	}
	return out
}

func (s *Session) Subscribe() (chan Event, func(), error) {
	ch := make(chan Event, 32)
	s.mu.Lock()
	if len(s.subs) >= maxSessionSubscribers {
		s.mu.Unlock()
		return nil, nil, errors.New("too many stream subscribers")
	}
	s.subs[ch] = struct{}{}
	s.mu.Unlock()
	return ch, func() {
		s.mu.Lock()
		delete(s.subs, ch)
		close(ch)
		s.mu.Unlock()
	}, nil
}

func (s *Session) setRunning(running bool) {
	s.mu.Lock()
	s.Running = running
	s.mu.Unlock()
}

func (s *Session) tryStart() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Running {
		return false
	}
	s.Running = true
	return true
}

func (srv *Server) initFixedSessions() {
	downloadCWD := srv.cfg.DataRoot
	if len(srv.cfg.DownloadDirs) > 0 {
		downloadCWD = srv.cfg.DownloadDirs[0]
	}
	downloader := NewSession("downloads", "下载器输出", downloadCWD, "fixed", true, false)
	downloader.MaxBytes = maxDownloadSessionBytes
	downloader.MaxEvents = maxDownloadSessionEvents
	downloader.Append("[downloader] fixed output session.")
	downloader.Append("[downloader] 下载、更新和代理切换日志会输出在这里。")
	srv.sessions.Add(downloader)

	logs := NewSession("logs", "容器日志", srv.cfg.AppRoot, "fixed", true, false)
	vllmLogs := NewSession("vllm-logs", "vLLM 日志", srv.cfg.AppRoot, "fixed", true, false)
	vllmLogs.Append("[vllm] logs session ready.")
	srv.sessions.Add(vllmLogs)
	logs.Append("[docker] logs session ready.")
	srv.sessions.Add(logs)
}

func (srv *Server) routes(mux *http.ServeMux) {
	mux.HandleFunc("/", srv.handleIndex)
	mux.HandleFunc("/api/health", srv.handleHealth)
	mux.HandleFunc("/api/config", srv.handleConfig)

	mux.HandleFunc("/api/container/status", srv.method("GET", srv.handleContainerStatus))
	mux.HandleFunc("/api/container/start", srv.method("POST", srv.handleContainerStart))
	mux.HandleFunc("/api/container/stop", srv.method("POST", srv.handleContainerStop))
	mux.HandleFunc("/api/container/restart", srv.method("POST", srv.handleContainerRestart))
	mux.HandleFunc("/api/container/vllm/logs/start", srv.method("POST", srv.handleVllmContainerLogsStart))
	mux.HandleFunc("/api/container/logs/start", srv.method("POST", srv.handleContainerLogsStart))
	mux.HandleFunc("/api/container/vllm/status", srv.method("GET", srv.handleVllmContainerStatus))
	mux.HandleFunc("/api/container/vllm/start", srv.method("POST", srv.handleVllmContainerStart))
	mux.HandleFunc("/api/container/vllm/stop", srv.method("POST", srv.handleVllmContainerStop))
	mux.HandleFunc("/api/container/vllm/service/start", srv.method("POST", srv.handleVllmServiceStart))
	mux.HandleFunc("/api/container/vllm/service/stop", srv.method("POST", srv.handleVllmServiceStop))

	mux.HandleFunc("/api/files/list", srv.method("GET", srv.handleFilesList))
	mux.HandleFunc("/api/files/delete", srv.method("POST", srv.handleFilesDelete))

	mux.HandleFunc("/api/download/dirs", srv.method("GET", srv.handleDownloadDirs))
	mux.HandleFunc("/api/terminal/sessions", srv.method("GET", srv.handleTerminalSessions))
	mux.HandleFunc("/api/terminal/create", srv.method("POST", srv.handleTerminalCreate))
	mux.HandleFunc("/api/terminal/vllm-docker-bash", srv.method("POST", srv.handleTerminalVllmDockerBash))
	mux.HandleFunc("/api/terminal/docker-bash", srv.method("POST", srv.handleTerminalDockerBash))
	mux.HandleFunc("/api/terminal/input", srv.method("POST", srv.handleTerminalInput))
	mux.HandleFunc("/api/terminal/control", srv.method("POST", srv.handleTerminalControl))
	mux.HandleFunc("/api/terminal/close", srv.method("POST", srv.handleTerminalClose))
	mux.HandleFunc("/api/terminal/stream", srv.method("GET", srv.handleTerminalStream))

	mux.HandleFunc("/api/download/start", srv.method("POST", srv.handleDownloadStart))
	mux.HandleFunc("/api/clash/state", srv.method("GET", srv.handleClashState))
	mux.HandleFunc("/api/clash/detect", srv.method("POST", srv.handleClashDetect))
	mux.HandleFunc("/api/clash/mode", srv.method("POST", srv.handleClashMode))
	mux.HandleFunc("/api/clash/proxy", srv.method("POST", srv.handleClashProxy))
	mux.HandleFunc("/api/clash/start", srv.method("POST", srv.handleClashStart))
	mux.HandleFunc("/api/clash/stop", srv.method("POST", srv.handleClashStop))

	mux.HandleFunc("/api/system/temp", srv.method("GET", srv.handleSystemTemp))
	mux.HandleFunc("/api/service/stop", srv.method("POST", srv.handleServiceStop))
	mux.HandleFunc("/api/service/restart", srv.method("POST", srv.handleServiceRestart))
}

func (srv *Server) method(method string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		next(w, r)
	}
}

func recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("panic recovered: %v", rec)
				http.Error(w, "internal server error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func withSecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(w, r)
	})
}

func localOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host, _, err := net.SplitHostPort(r.RemoteAddr)
		if err != nil || !isLoopbackHost(host) {
			http.Error(w, "local access only", http.StatusForbidden)
			return
		}
		origin := r.Header.Get("Origin")
		if origin != "" && !sameLocalOrigin(origin, r.Host) {
			http.Error(w, "origin rejected", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func loopbackBindHost(host string) string {
	host = strings.TrimSpace(host)
	if host == "" || !isLoopbackHost(host) {
		return "127.0.0.1"
	}
	return host
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func sameLocalOrigin(origin, hostport string) bool {
	u, err := http.NewRequest(http.MethodGet, origin, nil)
	if err != nil || u.URL == nil {
		return false
	}
	originHost := u.URL.Hostname()
	requestHost, _, err := net.SplitHostPort(hostport)
	if err != nil {
		requestHost = hostport
	}
	return isLoopbackHost(originHost) && isLoopbackHost(requestHost)
}

func (srv *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" && r.URL.Path != "/index.html" {
		http.NotFound(w, r)
		return
	}
	data, err := fs.ReadFile(staticFiles, "static/index.html")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(data)
}

func (srv *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "time": time.Now()})
}

func (srv *Server) handleConfig(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		cfg := srv.currentConfig()
		cfg.ConfigPath = "config.json"
		writeJSON(w, http.StatusOK, cfg)
	case http.MethodPost:
		cfg := srv.currentConfig()
		if err := readJSON(r, &cfg); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		normalizeConfig(&cfg)
		cfg.BindHost = loopbackBindHost(cfg.BindHost)
		if cfg.Port == 0 {
			cfg.Port = 8848
		}
		srv.cfgMu.Lock()
		srv.cfg = cfg
		srv.cfgMu.Unlock()
		if err := saveConfig("config.json", cfg); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, cfg)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (srv *Server) currentConfig() Config {
	srv.cfgMu.RLock()
	defer srv.cfgMu.RUnlock()
	return srv.cfg
}

func (srv *Server) handleContainerStatus(w http.ResponseWriter, r *http.Request) {
	status, err := srv.containerStatus(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": status})
}

func (srv *Server) handleContainerStart(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	if ok := srv.containerExists(r.Context(), name); !ok {
		writeError(w, http.StatusNotFound, fmt.Errorf("container %q not found", name))
		return
	}
	out, err := runCommand(r.Context(), "docker", "start", name)
	srv.appendLogs("logs", "[docker] docker start "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "running", "output": out})
}

func (srv *Server) handleContainerStop(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	out, err := runCommand(r.Context(), "docker", "stop", name)
	srv.appendLogs("logs", "[docker] docker stop "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "stopped", "output": out})
}

func (srv *Server) handleContainerRestart(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	out, err := runCommand(r.Context(), "docker", "restart", name)
	srv.appendLogs("logs", "[docker] docker restart "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "running", "output": out})
}

func (srv *Server) handleVllmContainerLogsStart(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().VllmContainerName
	vllmLogs, _ := srv.sessions.Get("vllm-logs")
	if vllmLogs == nil {
		writeError(w, http.StatusInternalServerError, errors.New("vllm logs session missing"))
		return
	}
	if !vllmLogs.tryStart() {
		writeJSON(w, http.StatusOK, map[string]string{"session": "vllm-logs", "status": "already-running"})
		return
	}
	vllmLogs.Append("[vllm] docker logs -f --tail=200 " + name)
	go srv.runStreamingCommandStarted(vllmLogs, srv.currentConfig().AppRoot, "docker", "logs", "-f", "--tail=200", name)
	writeJSON(w, http.StatusOK, map[string]string{"session": "vllm-logs"})
}

func (srv *Server) handleContainerLogsStart(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	logs, _ := srv.sessions.Get("logs")
	if logs == nil {
		writeError(w, http.StatusInternalServerError, errors.New("logs session missing"))
		return
	}
	if !logs.tryStart() {
		writeJSON(w, http.StatusOK, map[string]string{"session": "logs", "status": "already-running"})
		return
	}
	logs.Append("[docker] docker logs -f --tail=200 " + name)
	go srv.runStreamingCommandStarted(logs, srv.currentConfig().AppRoot, "docker", "logs", "-f", "--tail=200", name)
	writeJSON(w, http.StatusOK, map[string]string{"session": "logs"})
}

func (srv *Server) handleVllmContainerStatus(w http.ResponseWriter, r *http.Request) {
	status, err := srv.vllmContainerStatus(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	serviceStatus, err := srv.vllmServiceStatus(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": status, "serviceStatus": serviceStatus})
}

func (srv *Server) handleVllmContainerStart(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().VllmContainerName
	if ok := srv.containerExists(r.Context(), name); !ok {
		writeError(w, http.StatusNotFound, fmt.Errorf("container %q not found", name))
		return
	}
	out, err := runCommand(r.Context(), "docker", "start", name)
	srv.appendLogs("vllm-logs", "[vllm] docker start "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "running", "output": out})
}

func (srv *Server) handleVllmContainerStop(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().VllmContainerName
	out, err := runCommand(r.Context(), "docker", "stop", name)
	srv.appendLogs("vllm-logs", "[vllm] docker stop "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "stopped", "output": out})
}

func (srv *Server) handleVllmServiceStart(w http.ResponseWriter, r *http.Request) {
	cmd := srv.currentConfig().VllmServiceStartCommand
	if cmd == "" {
		writeError(w, http.StatusBadRequest, errors.New("vllmServiceStartCommand not configured"))
		return
	}
	containerStatus, err := srv.vllmContainerStatus(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if containerStatus != "running" {
		writeError(w, http.StatusConflict, errors.New("vllm container is not running"))
		return
	}
	if status, err := srv.vllmServiceStatus(r.Context()); err == nil && (status == "running" || status == "starting") {
		writeError(w, http.StatusConflict, errors.New("vllm service is already running"))
		return
	}
	srv.setVllmServiceStarting(true)
	if err := srv.runConfiguredActionDetached("vllm service start", cmd); err != nil {
		srv.setVllmServiceStarting(false)
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	go srv.watchVllmServiceStartup()
	writeJSON(w, http.StatusAccepted, map[string]string{"status": "starting"})
}

func (srv *Server) handleVllmServiceStop(w http.ResponseWriter, r *http.Request) {
	cmd := srv.currentConfig().VllmServiceStopCommand
	if cmd == "" {
		writeError(w, http.StatusBadRequest, errors.New("vllmServiceStopCommand not configured"))
		return
	}
	srv.setVllmServiceStarting(false)
	out, err := runCommand(r.Context(), "bash", "-c", cmd)
	srv.appendLogs("vllm-logs", "[vllm] service stop\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "stopped", "output": out})
}

func (srv *Server) vllmContainerStatus(ctx context.Context) (string, error) {
	name := srv.currentConfig().VllmContainerName
	if !srv.containerExists(ctx, name) {
		return "missing", nil
	}
	out, err := runCommand(ctx, "docker", "inspect", "--format={{.State.Status}}", name)
	if err != nil {
		return "", err
	}
	switch strings.TrimSpace(out) {
	case "running":
		return "running", nil
	default:
		return "stopped", nil
	}
}

func (srv *Server) vllmServiceStatus(ctx context.Context) (string, error) {
	if srv.isVllmServiceStarting() {
		return "starting", nil
	}
	return srv.probeVllmServiceStatus(ctx)
}

func (srv *Server) probeVllmServiceStatus(ctx context.Context) (string, error) {
	containerStatus, err := srv.vllmContainerStatus(ctx)
	if err != nil {
		return "", err
	}
	if containerStatus != "running" {
		return containerStatus, nil
	}
	name := srv.currentConfig().VllmContainerName
	port := srv.vllmServicePort()
	portHex := strings.ToUpper(fmt.Sprintf("%04X", port))
	probe := fmt.Sprintf("awk 'NR>1 { split($2,a,\":\"); if (toupper(a[2]) == \"%s\" && $4 == \"0A\") found=1 } END { exit(found ? 0 : 1) }' /proc/net/tcp /proc/net/tcp6", portHex)
	_, err = runCommandTimeout(ctx, 10*time.Second, "docker", "exec", name, "sh", "-lc", probe)
	if err != nil {
		return "stopped", nil
	}
	return "running", nil
}

func (srv *Server) vllmServicePort() int {
	command := srv.currentConfig().VllmServiceStartCommand
	port := 8000
	fields := strings.Fields(command)
	for _, field := range fields {
		if strings.HasPrefix(field, "--port=") {
			if next, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(field, "--port="))); err == nil && next > 0 {
				return next
			}
		}
	}
	for i, field := range fields {
		if field == "--port" && i+1 < len(fields) {
			if next, err := strconv.Atoi(strings.Trim(fields[i+1], "'\"")); err == nil && next > 0 {
				port = next
			}
			break
		}
	}
	return port
}

func (srv *Server) setVllmServiceStarting(starting bool) {
	srv.vllmMu.Lock()
	srv.vllmServiceStarting = starting
	srv.vllmMu.Unlock()
}

func (srv *Server) isVllmServiceStarting() bool {
	srv.vllmMu.RLock()
	defer srv.vllmMu.RUnlock()
	return srv.vllmServiceStarting
}

func (srv *Server) watchVllmServiceStartup() {
	defer srv.setVllmServiceStarting(false)
	deadline := time.NewTimer(60 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		if !srv.isVllmServiceStarting() {
			return
		}
		if status, err := srv.probeVllmServiceStatus(context.Background()); err == nil && status == "running" {
			return
		}
		select {
		case <-deadline.C:
			return
		case <-ticker.C:
		}
	}
}

func (srv *Server) containerExists(ctx context.Context, name string) bool {
	out, err := runCommand(ctx, "docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=^/"+name+"$")
	return err == nil && strings.TrimSpace(out) == name
}

func (srv *Server) containerStatus(ctx context.Context) (string, error) {
	name := srv.currentConfig().ContainerName
	if !srv.containerExists(ctx, name) {
		return "missing", nil
	}
	out, err := runCommand(ctx, "docker", "inspect", "--format={{.State.Status}}", name)
	if err != nil {
		return "", err
	}
	switch strings.TrimSpace(out) {
	case "running":
		return "running", nil
	default:
		return "stopped", nil
	}
}

func (srv *Server) appendLogs(sessionID, data string) {
	if s, ok := srv.sessions.Get(sessionID); ok {
		s.Append(data)
	}
}

func (srv *Server) handleFilesList(w http.ResponseWriter, r *http.Request) {
	path, _, _, err := srv.resolveManagedPath(r.URL.Query().Get("path"), true, true)
	if err != nil {
		writeError(w, http.StatusForbidden, err)
		return
	}
	entries, err := listDirectory(path)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	free, total := diskSpace(path)
	writeJSON(w, http.StatusOK, map[string]any{
		"path": path,
		"entries": entries,
		"freeBytes": free,
		"totalBytes": total,
		"roots": srv.managedRootViews(),
	})
}

func (srv *Server) handleFilesDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	containerPath, err := srv.hostToContainerPath(req.Path)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	name := srv.currentConfig().ContainerName
	out, err := runCommand(r.Context(), "docker", "exec", name, "rm", "-rf", "--", containerPath)
	srv.appendLogs("logs", "[docker] docker exec "+name+" rm -rf -- "+containerPath+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"deleted": req.Path, "containerPath": containerPath})
}

func (srv *Server) hostToContainerPath(path string) (string, error) {
	clean, root, rel, err := srv.resolveManagedPath(path, true, false)
	if err != nil {
		return "", err
	}
	if rel == "." {
		return "", errors.New("managed root directories cannot be deleted")
	}
	info, err := os.Lstat(clean)
	if err != nil {
		return "", err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return "", errors.New("symbolic links are not managed")
	}
	return pathJoinSlash(root.Container, filepath.ToSlash(rel)), nil
}

func listDirectory(path string) ([]FileEntry, error) {
	items, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	entries := make([]FileEntry, 0, min(len(items), maxFileEntries))
	for _, item := range items {
		if len(entries) >= maxFileEntries {
			break
		}
		name := item.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		info, err := item.Info()
		if err != nil {
			continue
		}
		kind := "FILE"
		if item.IsDir() {
			kind = "DIR"
		}
		entries = append(entries, FileEntry{
			Name: name,
			Path: filepath.Join(path, name),
			Type: kind,
			Size: info.Size(),
			ModTime: info.ModTime(),
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Type != entries[j].Type {
			return entries[i].Type == "DIR"
		}
		return strings.ToLower(entries[i].Name) < strings.ToLower(entries[j].Name)
	})
	return entries, nil
}

func filterManagedDownloadDirs(paths []string) []string {
	out := make([]string, 0, len(paths))
	seen := make(map[string]struct{}, len(paths))
	for _, raw := range paths {
		clean, err := filepath.Abs(strings.TrimSpace(raw))
		if err != nil || clean == "" {
			continue
		}
		if !pathInside(defaultDataRoot, clean) && !pathInside(defaultModelsRoot, clean) {
			continue
		}
		if hasHiddenRel(pathRelForFilter(defaultDataRoot, defaultModelsRoot, clean)) {
			continue
		}
		if _, ok := seen[clean]; ok {
			continue
		}
		seen[clean] = struct{}{}
		out = append(out, clean)
	}
	return out
}

func pathRelForFilter(rootA, rootB, clean string) string {
	if pathInside(rootA, clean) {
		rel, _ := filepath.Rel(rootA, clean)
		return rel
	}
	rel, _ := filepath.Rel(rootB, clean)
	return rel
}

func (srv *Server) managedRoots() []ManagedRoot {
	cfg := srv.currentConfig()
	return []ManagedRoot{
		{Name: "data", Host: defaultDataRoot, Container: strings.TrimRight(cfg.ContainerDataRoot, "/")},
		{Name: "models", Host: defaultModelsRoot, Container: pathJoinSlash(cfg.ContainerDataRoot, "models")},
	}
}

func (srv *Server) managedRootViews() []map[string]string {
	roots := srv.managedRoots()
	out := make([]map[string]string, 0, len(roots))
	for _, root := range roots {
		out = append(out, map[string]string{"name": root.Name, "path": root.Host})
	}
	return out
}

func (srv *Server) resolveManagedPath(raw string, mustExist, dirRequired bool) (string, ManagedRoot, string, error) {
	path := strings.TrimSpace(raw)
	if path == "" {
		path = defaultDataRoot
	}
	clean, err := filepath.Abs(path)
	if err != nil {
		return "", ManagedRoot{}, "", err
	}
	for _, root := range srv.managedRoots() {
		rootClean, err := filepath.Abs(root.Host)
		if err != nil {
			continue
		}
		if !pathInside(rootClean, clean) {
			continue
		}
		rel, err := filepath.Rel(rootClean, clean)
		if err != nil {
			return "", ManagedRoot{}, "", err
		}
		if hasHiddenRel(rel) {
			return "", ManagedRoot{}, "", errors.New("hidden paths are not managed")
		}
		if mustExist {
			if err := ensureNoSymlinkEscape(rootClean, clean); err != nil {
				return "", ManagedRoot{}, "", err
			}
			info, err := os.Stat(clean)
			if err != nil {
				return "", ManagedRoot{}, "", err
			}
			if dirRequired && !info.IsDir() {
				return "", ManagedRoot{}, "", fmt.Errorf("not a directory: %s", clean)
			}
		}
		return clean, root, rel, nil
	}
	return "", ManagedRoot{}, "", fmt.Errorf("only paths under %s or %s can be managed", defaultDataRoot, defaultModelsRoot)
}

func pathInside(root, target string) bool {
	rootClean, err := filepath.Abs(root)
	if err != nil {
		return false
	}
	targetClean, err := filepath.Abs(target)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(rootClean, targetClean)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel))
}

func hasHiddenRel(rel string) bool {
	if rel == "." || rel == "" {
		return false
	}
	for _, part := range strings.Split(filepath.Clean(rel), string(filepath.Separator)) {
		if strings.HasPrefix(part, ".") {
			return true
		}
	}
	return false
}

func ensureNoSymlinkEscape(root, target string) error {
	rootEval, err := filepath.EvalSymlinks(root)
	if err != nil {
		return err
	}
	targetEval, err := filepath.EvalSymlinks(target)
	if err != nil {
		return err
	}
	if !pathInside(rootEval, targetEval) {
		return fmt.Errorf("path escapes managed root through a symbolic link: %s", target)
	}
	return nil
}

func (srv *Server) handleDownloadDirs(w http.ResponseWriter, r *http.Request) {
	dirs, err := srv.listManagedDownloadDirs()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"dirs": dirs})
}

func (srv *Server) listManagedDownloadDirs() ([]DownloadDirEntry, error) {
	entries := make([]DownloadDirEntry, 0, 64)
	for _, root := range srv.managedRoots() {
		rootClean, err := filepath.Abs(root.Host)
		if err != nil {
			continue
		}
		info, err := os.Stat(rootClean)
		if err != nil || !info.IsDir() {
			continue
		}
		entries = append(entries, DownloadDirEntry{Path: rootClean, Label: root.Name})
		children, err := os.ReadDir(rootClean)
		if err != nil {
			return nil, err
		}
		for _, child := range children {
			if len(entries) >= maxDownloadDirEntries {
				break
			}
			name := child.Name()
			if strings.HasPrefix(name, ".") || !child.IsDir() {
				continue
			}
			entries = append(entries, DownloadDirEntry{
				Path: filepath.Join(rootClean, name),
				Label: root.Name + "/" + name,
			})
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		return strings.ToLower(entries[i].Label) < strings.ToLower(entries[j].Label)
	})
	return entries, nil
}

func safeWorkingDir(path, fallback string) (string, error) {
	if strings.TrimSpace(path) == "" {
		path = fallback
	}
	clean, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(clean)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("not a directory: %s", clean)
	}
	return clean, nil
}

func diskSpace(path string) (uint64, uint64) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, 0
	}
	return stat.Bavail * uint64(stat.Bsize), stat.Blocks * uint64(stat.Bsize)
}

func (srv *Server) handleTerminalSessions(w http.ResponseWriter, r *http.Request) {
	type sessionView struct {
		ID string `json:"id"`
		Title string `json:"title"`
		CWD string `json:"cwd"`
		Kind string `json:"kind"`
		Fixed bool `json:"fixed"`
		Closable bool `json:"closable"`
		Running bool `json:"running"`
	}
	list := srv.sessions.List()
	resp := make([]sessionView, 0, len(list))
	for _, s := range list {
		if s.ID == "downloads" {
			continue
		}
		s.mu.Lock()
		resp = append(resp, sessionView{s.ID, s.Title, s.CWD, s.Kind, s.Fixed, s.Closable, s.Running})
		s.mu.Unlock()
	}
	writeJSON(w, http.StatusOK, resp)
}

func (srv *Server) handleTerminalCreate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		CWD string `json:"cwd"`
	}
	_ = readJSON(r, &req)
	if req.CWD == "" {
		req.CWD = srv.currentConfig().DataRoot
	}
	session, err := srv.createShellSession(req.CWD, "命令行", []string{"/bin/bash", "-i"}, true, true)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": session.ID})
}

func (srv *Server) handleTerminalVllmDockerBash(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().VllmContainerName
	session, err := srv.createShellSession(srv.currentConfig().DataRoot, "vLLM bash", []string{"docker", "exec", "-it", name, "bash", "-i"}, true, false)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	session.mu.Lock()
	session.CWD = "/workspace"
	session.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"id": session.ID})
}

func (srv *Server) handleTerminalDockerBash(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	session, err := srv.createShellSession(srv.currentConfig().DataRoot, "Docker bash", []string{"docker", "exec", "-it", name, "bash", "-i"}, true, false)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	session.mu.Lock()
	session.CWD = "/opt/ComfyUI"
	session.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"id": session.ID})
}

func (srv *Server) createShellSession(cwd, title string, argv []string, managedCWD, trackCWD bool) (*Session, error) {
	if srv.sessions.UserCount() >= maxUserSessions {
		return nil, fmt.Errorf("最多只能同时打开 %d 个普通命令行", maxUserSessions)
	}
	if len(argv) == 0 {
		return nil, errors.New("empty command")
	}
	var checkedCWD string
	var err error
	if managedCWD {
		checkedCWD, _, _, err = srv.resolveManagedPath(cwd, true, true)
		if err != nil {
			return nil, err
		}
	} else {
		checkedCWD, err = safeWorkingDir(cwd, srv.currentConfig().DataRoot)
		if err != nil {
			return nil, err
		}
	}
	id := srv.sessions.NextUserID("term")
	session := NewSession(id, title+" "+id, checkedCWD, "user", false, true)
	session.trackCWD = trackCWD
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = checkedCWD
	cmd.Env = terminalEnv(os.Environ())
	if err := startPTYSession(session, cmd); err != nil {
		return nil, err
	}
	session.setRunning(true)
	session.Append("[pty] " + strings.Join(argv, " ") + "\n")
	srv.sessions.Add(session)
	return session, nil
}

func startPTYSession(session *Session, cmd *exec.Cmd) error {
	master, slave, err := openPTY()
	if err != nil {
		return err
	}
	cmd.Stdin = slave
	cmd.Stdout = slave
	cmd.Stderr = slave
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true, Setctty: true, Ctty: 0}
	if err := cmd.Start(); err != nil {
		_ = master.Close()
		_ = slave.Close()
		return err
	}
	_ = slave.Close()

	done := make(chan error, 1)
	session.mu.Lock()
	session.cmd = cmd
	session.stdin = master
	session.pty = master
	session.done = done
	session.mu.Unlock()

	go pipeToSession(session, master)
	go func() {
		err := cmd.Wait()
		session.setRunning(false)
		_ = master.Close()
		if err != nil {
			session.Append("\n[process exited] " + err.Error() + "\n")
		} else {
			session.Append("\n[process exited] ok\n")
		}
		done <- err
	}()
	return nil
}

func openPTY() (*os.File, *os.File, error) {
	masterFD, err := syscall.Open("/dev/ptmx", syscall.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		return nil, nil, err
	}
	master := os.NewFile(uintptr(masterFD), "/dev/ptmx")
	unlock := int32(0)
	if err := ioctl(master.Fd(), tiocsptlck, uintptr(unsafe.Pointer(&unlock))); err != nil {
		_ = master.Close()
		return nil, nil, err
	}
	var ptyNumber uint32
	if err := ioctl(master.Fd(), tiocgptn, uintptr(unsafe.Pointer(&ptyNumber))); err != nil {
		_ = master.Close()
		return nil, nil, err
	}
	slaveName := fmt.Sprintf("/dev/pts/%d", ptyNumber)
	slaveFD, err := syscall.Open(slaveName, syscall.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		_ = master.Close()
		return nil, nil, err
	}
	slave := os.NewFile(uintptr(slaveFD), slaveName)
	return master, slave, nil
}

func ioctl(fd uintptr, req uintptr, arg uintptr) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, fd, req, arg)
	if errno != 0 {
		return errno
	}
	return nil
}

func terminalEnv(base []string) []string {
	env := make([]string, 0, len(base)+6)
	blocked := map[string]struct{}{
		"TERM": {},
		"COLORTERM": {},
		"CLICOLOR": {},
		"CLICOLOR_FORCE": {},
		"FORCE_COLOR": {},
	}
	for _, item := range base {
		key := item
		if i := strings.IndexByte(item, '='); i >= 0 {
			key = item[:i]
		}
		if _, ok := blocked[key]; ok {
			continue
		}
		env = append(env, item)
	}
	env = append(env,
		"TERM=xterm-256color",
		"COLORTERM=truecolor",
		"CLICOLOR=1",
		"CLICOLOR_FORCE=1",
		"FORCE_COLOR=1",
	)
	return env
}

func (srv *Server) handleTerminalInput(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
		Data string `json:"data"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if len(req.Data) > maxTerminalInputBytes {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Errorf("terminal input exceeds %d bytes", maxTerminalInputBytes))
		return
	}
	session, ok := srv.sessions.Get(req.ID)
	if !ok {
		writeError(w, http.StatusNotFound, errors.New("session not found"))
		return
	}
	session.mu.Lock()
	stdin := session.stdin
	session.mu.Unlock()
	if stdin == nil {
		writeError(w, http.StatusBadRequest, errors.New("session is not writable"))
		return
	}
	_, err := io.WriteString(stdin, req.Data+"\n")
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	cwd := refreshProcessCWD(session, 80*time.Millisecond)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "cwd": cwd})
}

func refreshProcessCWD(session *Session, settle time.Duration) string {
	session.mu.Lock()
	cwd := session.CWD
	track := session.trackCWD
	cmd := session.cmd
	session.mu.Unlock()
	if !track || cmd == nil || cmd.Process == nil {
		return cwd
	}
	if settle > 0 {
		time.Sleep(settle)
	}
	target, err := os.Readlink("/proc/" + strconv.Itoa(cmd.Process.Pid) + "/cwd")
	if err != nil || target == "" {
		return cwd
	}
	clean, err := filepath.Abs(target)
	if err != nil {
		return cwd
	}
	session.mu.Lock()
	session.CWD = clean
	session.mu.Unlock()
	return clean
}

func (srv *Server) handleTerminalControl(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
		Control string `json:"control"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	session, ok := srv.sessions.Get(req.ID)
	if !ok {
		writeError(w, http.StatusNotFound, errors.New("session not found"))
		return
	}
	if err := sendControl(session, req.Control); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func sendControl(session *Session, control string) error {
	session.mu.Lock()
	cmd := session.cmd
	stdin := session.stdin
	session.mu.Unlock()

	if strings.HasPrefix(strings.ToUpper(strings.TrimSpace(control)), "CTRL+L") {
		session.Clear()
		if stdin != nil {
			_, _ = io.WriteString(stdin, "\x0c")
		}
		return nil
	}
	if data, ok := controlBytes(control); ok {
		if stdin != nil {
			_, err := stdin.Write(data)
			if err == nil {
				return nil
			}
		}
		if cmd != nil && cmd.Process != nil && len(data) == 1 {
			switch data[0] {
			case 0x03:
				return syscall.Kill(-cmd.Process.Pid, syscall.SIGINT)
			case 0x1a:
				return syscall.Kill(-cmd.Process.Pid, syscall.SIGTSTP)
			}
		}
	}
	if cmd == nil || cmd.Process == nil {
		session.Append("[control] " + control + "\n")
		return nil
	}
	switch {
	case strings.HasPrefix(control, "Ctrl+C"):
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGINT)
	case strings.HasPrefix(control, "Ctrl+Z"):
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGTSTP)
	case strings.HasPrefix(control, "Ctrl+D"):
		if stdin != nil {
			return stdin.Close()
		}
	case strings.HasPrefix(control, "Tab"):
		if stdin != nil {
			_, err := io.WriteString(stdin, "\t")
			return err
		}
	case strings.HasPrefix(control, "Esc"):
		if stdin != nil {
			_, err := io.WriteString(stdin, "\x1b")
			return err
		}
	}
	session.Append("[control] " + control + "\n")
	return nil
}

func controlBytes(control string) ([]byte, bool) {
	name := strings.ToUpper(strings.TrimSpace(control))
	if i := strings.IndexAny(name, " \t"); i >= 0 {
		name = name[:i]
	}
	switch name {
	case "TAB":
		return []byte{'\t'}, true
	case "ESC", "ESCAPE":
		return []byte{0x1b}, true
	case "ENTER", "RETURN":
		return []byte{'\r'}, true
	}
	if !strings.HasPrefix(name, "CTRL+") {
		return nil, false
	}
	key := strings.TrimPrefix(name, "CTRL+")
	if key == "" {
		return nil, false
	}
	switch key {
	case "SPACE":
		return []byte{0x00}, true
	case "[":
		return []byte{0x1b}, true
	case "\\":
		return []byte{0x1c}, true
	case "]":
		return []byte{0x1d}, true
	case "^":
		return []byte{0x1e}, true
	case "_":
		return []byte{0x1f}, true
	}
	ch := key[0]
	if ch >= 'A' && ch <= 'Z' {
		return []byte{ch - 'A' + 1}, true
	}
	return nil, false
}

func (srv *Server) handleTerminalClose(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	session, ok := srv.sessions.Get(req.ID)
	if !ok {
		writeJSON(w, http.StatusOK, map[string]bool{"closed": true})
		return
	}
	if session.Fixed {
		writeError(w, http.StatusBadRequest, errors.New("fixed sessions cannot be closed"))
		return
	}
	if err := gracefulClose(session); err != nil {
		writeError(w, http.StatusConflict, err)
		return
	}
	srv.sessions.Remove(req.ID)
	writeJSON(w, http.StatusOK, map[string]bool{"closed": true})
}

func gracefulClose(session *Session) error {
	session.mu.Lock()
	cmd := session.cmd
	running := session.Running
	done := session.done
	session.mu.Unlock()
	if cmd == nil || cmd.Process == nil || !running {
		return nil
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	select {
	case <-done:
		return nil
	case <-time.After(2 * time.Second):
		return errors.New("process did not exit after SIGTERM; session kept open")
	}
}

func (srv *Server) shutdownSessions() {
	for _, session := range srv.sessions.List() {
		if err := gracefulClose(session); err != nil {
			session.Append("[shutdown] " + err.Error())
		}
	}
}

func (srv *Server) handleTerminalStream(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	session, ok := srv.sessions.Get(id)
	if !ok {
		writeError(w, http.StatusNotFound, errors.New("session not found"))
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, errors.New("streaming unsupported"))
		return
	}
	lastID, _ := strconv.ParseUint(r.Header.Get("Last-Event-ID"), 10, 64)
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Connection", "keep-alive")

	for _, ev := range session.Snapshot(lastID) {
		writeSSE(w, ev)
	}
	flusher.Flush()

	ch, cancel, err := session.Subscribe()
	if err != nil {
		writeError(w, http.StatusTooManyRequests, err)
		return
	}
	defer cancel()
	ticker := time.NewTicker(20 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case ev := <-ch:
			writeSSE(w, ev)
			flusher.Flush()
		case <-ticker.C:
			_, _ = io.WriteString(w, ": keepalive\n\n")
			flusher.Flush()
		}
	}
}

func (srv *Server) handleDownloadStart(w http.ResponseWriter, r *http.Request) {
	var req DownloadTask
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	req.Method = strings.TrimSpace(strings.ToLower(req.Method))
	if req.Dir == "" {
		writeError(w, http.StatusBadRequest, errors.New("download dir required"))
		return
	}
	if (req.Method == "wget" || req.Method == "git clone") && req.URL == "" {
		writeError(w, http.StatusBadRequest, errors.New("download url required"))
		return
	}
	if req.Method != "wget" && req.Method != "git clone" && req.Method != "git pull" {
		writeError(w, http.StatusBadRequest, errors.New("method must be wget, git clone, or git pull"))
		return
	}
	dir, _, _, err := srv.resolveManagedPath(req.Dir, true, true)
	if err != nil {
		writeError(w, http.StatusForbidden, err)
		return
	}
	req.Dir = dir
	req.ID = time.Now().UnixNano()
	select {
	case srv.downloads <- req:
	default:
		writeError(w, http.StatusTooManyRequests, errors.New("download queue is full"))
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"queued": true, "id": req.ID})
}

func (srv *Server) downloadWorker() {
	for task := range srv.downloads {
		session, _ := srv.sessions.Get("downloads")
		if session == nil {
			continue
		}
		session.mu.Lock()
		session.CWD = task.Dir
		session.mu.Unlock()
		session.Append(fmt.Sprintf("[download #%d] %s", task.ID, task.Method))
		if _, _, _, err := srv.resolveManagedPath(task.Dir, true, true); err != nil {
			session.Append("[error] " + err.Error())
			continue
		}
		var cmd *exec.Cmd
		switch task.Method {
		case "wget":
			cmd = exec.Command("wget", task.URL)
		case "git clone":
			cmd = exec.Command("git", "clone", task.URL)
		case "git pull":
			cmd = exec.Command("git", "pull")
		}
		cmd.Dir = task.Dir
		if task.UseProxy || srv.isDownloadProxyEnabled() {
			cmd.Env = append(os.Environ(), srv.proxyEnv()...)
		}
		srv.runCommandToSession(session, cmd)
	}
}

func (srv *Server) isDownloadProxyEnabled() bool {
	srv.dlMu.RLock()
	defer srv.dlMu.RUnlock()
	return srv.downloadProxy
}

func (srv *Server) proxyEnv() []string {
	cfg := srv.currentConfig()
	return []string{
		"HTTP_PROXY=" + cfg.HTTPProxy,
		"HTTPS_PROXY=" + cfg.HTTPProxy,
		"ALL_PROXY=" + cfg.SocksProxy,
	}
}

func (srv *Server) handleClashState(w http.ResponseWriter, r *http.Request) {
	systemProxy, systemProxyChecked := srv.detectSystemProxy()
	srv.clashMu.RLock()
	state := map[string]any{
		"mode": srv.clashMode,
		"apiDetected": srv.clashAPI,
		"apiChecked": srv.clashAPIChecked,
		"appRunning": srv.clashRunning,
		"appChecked": srv.clashRunningChecked,
		"systemProxy": systemProxy,
		"systemProxyChecked": systemProxyChecked,
	}
	srv.clashMu.RUnlock()
	state["downloadProxy"] = srv.isDownloadProxyEnabled()
	writeJSON(w, http.StatusOK, state)
}

func (srv *Server) handleClashDetect(w http.ResponseWriter, r *http.Request) {
	base, _, _, err := srv.clashAPIRequest(r.Context(), http.MethodGet, "/version", nil)
	ok := err == nil
	mode := srv.currentClashMode()
	if ok {
		if _, _, data, cfgErr := srv.clashAPIRequest(r.Context(), http.MethodGet, "/configs", nil); cfgErr == nil {
			if nextMode := parseClashMode(data); nextMode != "" {
				mode = nextMode
			}
		}
	}
	srv.clashMu.Lock()
	srv.clashAPI = ok
	srv.clashAPIChecked = true
	srv.clashMode = mode
	if ok {
		srv.clashRunning = true
		srv.clashRunningChecked = true
	}
	srv.clashMu.Unlock()
	if downloads, ok := srv.sessions.Get("downloads"); ok {
		if err != nil {
			downloads.Append("[clash] detect failed: " + err.Error())
		} else {
			downloads.Append(fmt.Sprintf("[clash] detect api: %v url=%s mode=%s\n", ok, base, mode))
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": ok, "url": base, "mode": mode, "error": errString(err)})
}

func (srv *Server) handleClashMode(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Mode string `json:"mode"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	mode, apiMode, ok := normalizeMode(req.Mode)
	if !ok {
		writeError(w, http.StatusBadRequest, errors.New("mode must be rule, global, or direct"))
		return
	}
	cfg := srv.currentConfig()
	body, _ := json.Marshal(map[string]string{"mode": apiMode})
	base, _, _, err := srv.clashAPIRequest(r.Context(), http.MethodPatch, "/configs", body)
	apiOK := err == nil
	commandOK := false
	command := clashModeCommand(cfg, mode)
	if !apiOK && command != "" {
		if cmdErr := srv.runConfiguredActionSync(r.Context(), "Clash Verge "+clashModeName(mode), command, 8*time.Second); cmdErr == nil {
			err = nil
			commandOK = true
		} else {
			err = fmt.Errorf("api: %v; command: %w", err, cmdErr)
		}
	}
	srv.clashMu.Lock()
	srv.clashAPI = apiOK
	srv.clashAPIChecked = true
	if apiOK || commandOK {
		srv.clashMode = mode
		if apiOK {
			srv.clashRunning = true
			srv.clashRunningChecked = true
		}
	}
	srv.clashMu.Unlock()
	if downloads, ok := srv.sessions.Get("downloads"); ok {
		downloads.Append(fmt.Sprintf("[clash] PATCH %s/configs mode=%s apiOK=%v commandOK=%v error=%s\n", base, apiMode, apiOK, commandOK, errString(err)))
	}
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"mode": mode, "apiOK": apiOK, "commandOK": commandOK})
}

func (srv *Server) handleClashProxy(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Enabled bool `json:"enabled"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if err := srv.setSystemProxy(r.Context(), req.Enabled); err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	srv.dlMu.Lock()
	srv.downloadProxy = req.Enabled
	srv.dlMu.Unlock()
	if downloads, ok := srv.sessions.Get("downloads"); ok {
		downloads.Append(fmt.Sprintf("[proxy] system proxy enabled=%v; download proxy env enabled=%v\n", req.Enabled, req.Enabled))
	}
	writeJSON(w, http.StatusOK, map[string]bool{"enabled": req.Enabled, "systemProxy": req.Enabled, "downloadProxy": req.Enabled})
}

func (srv *Server) handleClashStart(w http.ResponseWriter, r *http.Request) {
	cfg := srv.currentConfig()
	if cfg.ClashStartCommand == "" {
		writeError(w, http.StatusBadRequest, errors.New("clash start command is empty"))
		return
	}
	go srv.runConfiguredAction("启动 Clash Verge", cfg.ClashStartCommand, true)
	srv.clashMu.Lock()
	srv.clashRunning = true
	srv.clashRunningChecked = true
	srv.clashMu.Unlock()
	writeJSON(w, http.StatusAccepted, map[string]bool{"started": true})
}

func (srv *Server) handleClashStop(w http.ResponseWriter, r *http.Request) {
	cfg := srv.currentConfig()
	if cfg.ClashStopCommand == "" {
		writeError(w, http.StatusBadRequest, errors.New("clash stop command is empty"))
		return
	}
	go srv.runConfiguredAction("停止 Clash Verge", cfg.ClashStopCommand, false)
	srv.clashMu.Lock()
	srv.clashRunning = false
	srv.clashRunningChecked = true
	srv.clashMu.Unlock()
	writeJSON(w, http.StatusAccepted, map[string]bool{"stopped": true})
}

func (srv *Server) handleSystemTemp(w http.ResponseWriter, r *http.Request) {
	type TempZone struct {
		Name string  `json:"name"`
		Temp float64 `json:"temp"`
	}
	zones := make([]TempZone, 0, 8)
	entries, err := os.ReadDir("/sys/class/thermal")
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"zones": zones, "error": err.Error()})
		return
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, "thermal_zone") {
			continue
		}
		typePath := "/sys/class/thermal/" + name + "/type"
		tempPath := "/sys/class/thermal/" + name + "/temp"
		typeBytes, err := os.ReadFile(typePath)
		if err != nil {
			continue
		}
		tempBytes, err := os.ReadFile(tempPath)
		if err != nil {
			continue
		}
		milliC, _ := strconv.ParseInt(strings.TrimSpace(string(tempBytes)), 10, 64)
		zones = append(zones, TempZone{
			Name: strings.TrimSpace(string(typeBytes)),
			Temp: float64(milliC) / 1000.0,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"zones": zones})
}

func (srv *Server) handleServiceStop(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusAccepted, map[string]bool{"stopping": true})
	go srv.stopService(false)
}

func (srv *Server) handleServiceRestart(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusAccepted, map[string]bool{"restarting": true})
	go srv.stopService(true)
}

func (srv *Server) stopService(restart bool) {
	srv.serviceOnce.Do(func() {
		time.Sleep(350 * time.Millisecond)
		if restart {
			if err := startSelfAfterDelay(); err != nil {
				log.Printf("self restart failed: %v", err)
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.shutdownSessions()
		if srv.httpServer != nil {
			_ = srv.httpServer.Shutdown(ctx)
		}
		os.Exit(0)
	})
}

func startSelfAfterDelay() error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	wd, _ := os.Getwd()
	args := append([]string{"-c", "sleep 0.6; exec \"$0\" \"$@\"", exe}, os.Args[1:]...)
	cmd := exec.Command("/bin/sh", args...)
	cmd.Dir = wd
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return err
	}
	return cmd.Process.Release()
}

func normalizeMode(mode string) (string, string, bool) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "rule", "rules":
		return "rule", "Rule", true
	case "global":
		return "global", "Global", true
	case "direct":
		return "direct", "Direct", true
	default:
		return "", "", false
	}
}

func clashModeCommand(cfg Config, mode string) string {
	switch mode {
	case "rule":
		return strings.TrimSpace(cfg.ClashRuleModeCommand)
	case "global":
		return strings.TrimSpace(cfg.ClashGlobalModeCommand)
	case "direct":
		return strings.TrimSpace(cfg.ClashDirectModeCommand)
	default:
		return ""
	}
}

func clashModeName(mode string) string {
	switch mode {
	case "rule":
		return "规则模式"
	case "global":
		return "全局模式"
	case "direct":
		return "直连模式"
	default:
		return mode
	}
}

func (srv *Server) currentClashMode() string {
	srv.clashMu.RLock()
	defer srv.clashMu.RUnlock()
	if srv.clashMode == "" {
		return "rule"
	}
	return srv.clashMode
}

func parseClashMode(data []byte) string {
	var cfg struct {
		Mode string `json:"mode"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}
	mode, _, ok := normalizeMode(cfg.Mode)
	if !ok {
		return ""
	}
	return mode
}

func (srv *Server) clashControllerURLs() []string {
	cfg := srv.currentConfig()
	candidates := []string{
		strings.TrimRight(cfg.ClashControllerURL, "/"),
		"http://127.0.0.1:9090",
		"http://127.0.0.1:9097",
		"http://127.0.0.1:9091",
	}
	out := make([]string, 0, len(candidates))
	seen := make(map[string]struct{}, len(candidates))
	for _, item := range candidates {
		if item == "" {
			continue
		}
		if _, ok := seen[item]; ok {
			continue
		}
		seen[item] = struct{}{}
		out = append(out, item)
	}
	return out
}

func (srv *Server) clashAPIRequest(ctx context.Context, method, apiPath string, body []byte) (string, int, []byte, error) {
	cfg := srv.currentConfig()
	var lastErr error
	for _, base := range srv.clashControllerURLs() {
		req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(base, "/")+apiPath, bytes.NewReader(body))
		if err != nil {
			lastErr = err
			continue
		}
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		if cfg.ClashSecret != "" {
			req.Header.Set("Authorization", "Bearer "+cfg.ClashSecret)
		}
		client := http.Client{Timeout: 3 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		data, readErr := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		_ = resp.Body.Close()
		if readErr != nil {
			lastErr = readErr
			continue
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			srv.rememberClashURL(base)
			return base, resp.StatusCode, data, nil
		}
		lastErr = fmt.Errorf("%s %s returned HTTP %d: %s", method, base+apiPath, resp.StatusCode, strings.TrimSpace(string(data)))
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			break
		}
	}
	if lastErr == nil {
		lastErr = errors.New("clash controller api is not reachable")
	}
	return "", 0, nil, lastErr
}

func (srv *Server) rememberClashURL(base string) {
	cfg := srv.currentConfig()
	if strings.TrimRight(cfg.ClashControllerURL, "/") == base {
		return
	}
	cfg.ClashControllerURL = base
	srv.cfgMu.Lock()
	srv.cfg.ClashControllerURL = base
	srv.cfgMu.Unlock()
}

func (srv *Server) setSystemProxy(ctx context.Context, enabled bool) error {
	cfg := srv.currentConfig()
	command := strings.TrimSpace(cfg.ClashSystemProxyOffCommand)
	title := "关闭 Clash Verge 系统代理"
	if enabled {
		command = strings.TrimSpace(cfg.ClashSystemProxyOnCommand)
		title = "开启 Clash Verge 系统代理"
	}
	if command != "" {
		return srv.runConfiguredActionSync(ctx, title, command, 8*time.Second)
	}

	var failures []string
	if err := srv.setGSettingsProxy(ctx, enabled); err != nil {
		failures = append(failures, "gsettings: "+err.Error())
	}
	if err := srv.patchClashVergeSystemProxy(enabled); err != nil {
		failures = append(failures, "verge.yaml: "+err.Error())
	}
	if len(failures) == 2 {
		return errors.New(strings.Join(failures, "; "))
	}
	return nil
}

func (srv *Server) setGSettingsProxy(ctx context.Context, enabled bool) error {
	if _, err := exec.LookPath("gsettings"); err != nil {
		return err
	}
	if !enabled {
		_, err := runCommandTimeout(ctx, 5*time.Second, "gsettings", "set", "org.gnome.system.proxy", "mode", "none")
		return err
	}
	cfg := srv.currentConfig()
	httpHost, httpPort, err := proxyHostPort(cfg.HTTPProxy, "127.0.0.1", 7890)
	if err != nil {
		return err
	}
	socksHost, socksPort, err := proxyHostPort(cfg.SocksProxy, httpHost, 7891)
	if err != nil {
		return err
	}
	commands := [][]string{
		{"gsettings", "set", "org.gnome.system.proxy.http", "host", httpHost},
		{"gsettings", "set", "org.gnome.system.proxy.http", "port", strconv.Itoa(httpPort)},
		{"gsettings", "set", "org.gnome.system.proxy.https", "host", httpHost},
		{"gsettings", "set", "org.gnome.system.proxy.https", "port", strconv.Itoa(httpPort)},
		{"gsettings", "set", "org.gnome.system.proxy.socks", "host", socksHost},
		{"gsettings", "set", "org.gnome.system.proxy.socks", "port", strconv.Itoa(socksPort)},
		{"gsettings", "set", "org.gnome.system.proxy", "ignore-hosts", "['localhost','127.0.0.0/8','::1']"},
		{"gsettings", "set", "org.gnome.system.proxy", "mode", "manual"},
	}
	for _, argv := range commands {
		if _, err := runCommandTimeout(ctx, 5*time.Second, argv[0], argv[1:]...); err != nil {
			return err
		}
	}
	return nil
}

func (srv *Server) detectSystemProxy() (bool, bool) {
	if _, err := exec.LookPath("gsettings"); err == nil {
		out, err := runCommandTimeout(context.Background(), 2*time.Second, "gsettings", "get", "org.gnome.system.proxy", "mode")
		if err == nil {
			mode := strings.Trim(out, " \n\r\t'")
			if mode == "manual" || mode == "auto" {
				return true, true
			}
			if mode == "none" {
				return false, true
			}
		}
	}
	enabled, err := srv.readClashVergeSystemProxy()
	if err == nil {
		return enabled, true
	}
	return false, false
}

func proxyHostPort(raw, fallbackHost string, fallbackPort int) (string, int, error) {
	if strings.TrimSpace(raw) == "" {
		return fallbackHost, fallbackPort, nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", 0, err
	}
	host := u.Hostname()
	if host == "" {
		host = fallbackHost
	}
	port := fallbackPort
	if u.Port() != "" {
		parsed, err := strconv.Atoi(u.Port())
		if err != nil {
			return "", 0, err
		}
		port = parsed
	}
	return host, port, nil
}

func (srv *Server) patchClashVergeSystemProxy(enabled bool) error {
	var lastErr error
	for _, path := range srv.clashVergeConfigCandidates() {
		data, err := os.ReadFile(path)
		if err != nil {
			lastErr = err
			continue
		}
		next := patchSimpleYAMLBool(string(data), "enable_system_proxy", enabled)
		return os.WriteFile(path, []byte(next), 0600)
	}
	if lastErr == nil {
		lastErr = errors.New("verge config path is empty")
	}
	return lastErr
}

func (srv *Server) readClashVergeSystemProxy() (bool, error) {
	var lastErr error
	for _, path := range srv.clashVergeConfigCandidates() {
		data, err := os.ReadFile(path)
		if err != nil {
			lastErr = err
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "enable_system_proxy:") {
				value := strings.TrimSpace(strings.TrimPrefix(trimmed, "enable_system_proxy:"))
				return strings.EqualFold(value, "true"), nil
			}
		}
		lastErr = errors.New("enable_system_proxy not found")
	}
	if lastErr == nil {
		lastErr = errors.New("verge config path is empty")
	}
	return false, lastErr
}

func (srv *Server) clashVergeConfigCandidates() []string {
	cfg := srv.currentConfig()
	candidates := []string{
		cfg.ClashVergeConfigPath,
		defaultClashVergeConfig,
		"~/.config/clash-verge-rev/verge.yaml",
		"~/.config/clash-verge/verge.yaml",
	}
	out := make([]string, 0, len(candidates))
	seen := make(map[string]struct{}, len(candidates))
	for _, item := range candidates {
		item = expandHome(strings.TrimSpace(item))
		if item == "" {
			continue
		}
		if _, ok := seen[item]; ok {
			continue
		}
		seen[item] = struct{}{}
		out = append(out, item)
	}
	return out
}

func patchSimpleYAMLBool(data, key string, value bool) string {
	lines := strings.Split(data, "\n")
	valueText := "false"
	if value {
		valueText = "true"
	}
	found := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, key+":") {
			prefixLen := len(line) - len(strings.TrimLeft(line, " \t"))
			lines[i] = line[:prefixLen] + key + ": " + valueText
			found = true
		}
	}
	if !found {
		if len(lines) > 0 && lines[len(lines)-1] == "" {
			lines[len(lines)-1] = key + ": " + valueText
			lines = append(lines, "")
		} else {
			lines = append(lines, key+": "+valueText)
		}
	}
	return strings.Join(lines, "\n")
}

func expandHome(path string) string {
	if path == "~" {
		if home, err := os.UserHomeDir(); err == nil {
			return home
		}
	}
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}

func (srv *Server) runConfiguredAction(title, command string, detach bool) {
	session, _ := srv.sessions.Get("downloads")
	if session == nil {
		return
	}
	session.Append("[" + title + "] " + command)
	argv, err := parseCommandLine(command)
	if err != nil {
		session.Append("[error] " + err.Error())
		return
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	if detach {
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		if err := cmd.Start(); err != nil {
			session.Append("[error] " + err.Error())
			return
		}
		session.Append(fmt.Sprintf("[started] pid=%d", cmd.Process.Pid))
		_ = cmd.Process.Release()
		return
	}
	srv.runCommandToSession(session, cmd)
}

func (srv *Server) runConfiguredActionSync(ctx context.Context, title, command string, timeout time.Duration) error {
	session, _ := srv.sessions.Get("downloads")
	if session != nil {
		session.Append("[" + title + "] " + command)
	}
	argv, err := parseCommandLine(command)
	if err != nil {
		if session != nil {
			session.Append("[error] " + err.Error())
		}
		return err
	}
	if len(argv) == 0 {
		err := errors.New("empty command")
		if session != nil {
			session.Append("[error] " + err.Error())
		}
		return err
	}
	out, err := runCommandTimeout(ctx, timeout, argv[0], argv[1:]...)
	if session != nil {
		if strings.TrimSpace(out) != "" {
			session.Append(out)
		}
		if err != nil {
			session.Append("[error] " + err.Error())
		} else {
			session.Append("[ok] command completed")
		}
	}
	return err
}

func (srv *Server) runConfiguredActionDetached(title, command string) error {
	session, _ := srv.sessions.Get("vllm-logs")
	if session != nil {
		session.Append("[" + title + "] " + command)
	}
	cmd := exec.Command("bash", "-lc", command)
	cmd.Dir = srv.currentConfig().AppRoot
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		if session != nil {
			session.Append("[error] " + err.Error())
		}
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		if session != nil {
			session.Append("[error] " + err.Error())
		}
		return err
	}
	if err := cmd.Start(); err != nil {
		if session != nil {
			session.Append("[error] " + err.Error())
		}
		return err
	}
	if session != nil {
		session.Append(fmt.Sprintf("[started] pid=%d", cmd.Process.Pid))
		go pipeToSession(session, stdout)
		go pipeToSession(session, stderr)
	} else {
		go io.Copy(io.Discard, stdout)
		go io.Copy(io.Discard, stderr)
	}
	go func() {
		err := cmd.Wait()
		if session == nil {
			if strings.Contains(strings.ToLower(title), "vllm service start") {
				srv.setVllmServiceStarting(false)
			}
			return
		}
		if err != nil {
			session.Append("[exit] " + err.Error())
		} else {
			session.Append("[exit] ok")
		}
		if strings.Contains(strings.ToLower(title), "vllm service start") {
			srv.setVllmServiceStarting(false)
		}
	}()
	return nil
}

func (srv *Server) runStreamingCommand(session *Session, cwd, name string, args ...string) {
	cmd := exec.Command(name, args...)
	cmd.Dir = cwd
	srv.runCommandToSession(session, cmd)
}

func (srv *Server) runStreamingCommandStarted(session *Session, cwd, name string, args ...string) {
	cmd := exec.Command(name, args...)
	cmd.Dir = cwd
	srv.runCommandToSessionWithState(session, cmd, true)
}

func (srv *Server) runCommandToSession(session *Session, cmd *exec.Cmd) {
	srv.runCommandToSessionWithState(session, cmd, false)
}

func (srv *Server) runCommandToSessionWithState(session *Session, cmd *exec.Cmd, alreadyRunning bool) {
	if !alreadyRunning {
		session.setRunning(true)
	}
	defer session.setRunning(false)
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}
	done := make(chan error, 1)
	session.mu.Lock()
	session.cmd = cmd
	session.done = done
	session.mu.Unlock()
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		session.Append("[error] " + err.Error())
		done <- err
		return
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		session.Append("[error] " + err.Error())
		done <- err
		return
	}
	if err := cmd.Start(); err != nil {
		session.Append("[error] " + err.Error())
		done <- err
		return
	}
	go pipeToSession(session, stdout)
	go pipeToSession(session, stderr)
	err = cmd.Wait()
	if err != nil {
		session.Append("[exit] " + err.Error())
	} else {
		session.Append("[exit] ok")
	}
	done <- err
}

func pipeToSession(session *Session, r io.Reader) {
	reader := bufio.NewReaderSize(r, 4096)
	buf := make([]byte, 4096)
	for {
		n, err := reader.Read(buf)
		if n > 0 {
			session.Append(string(buf[:n]))
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				if pathErr, ok := err.(*os.PathError); ok && errors.Is(pathErr.Err, syscall.EIO) {
					return
				}
				session.Append("[read error] " + err.Error())
			}
			return
		}
	}
}

func runCommand(ctx context.Context, name string, args ...string) (string, error) {
	return runCommandTimeout(ctx, 30*time.Second, name, args...)
}

func runCommandTimeout(ctx context.Context, timeout time.Duration, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	if out.Len() > 64*1024 {
		data := out.Bytes()
		return string(data[:64*1024]) + "\n[output truncated]", err
	}
	return out.String(), err
}

func readJSON(r *http.Request, dst any) error {
	defer r.Body.Close()
	dec := json.NewDecoder(io.LimitReader(r.Body, requestBodyLimit))
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeSSE(w io.Writer, ev Event) {
	_, _ = fmt.Fprintf(w, "id: %d\n", ev.Seq)
	data := strings.ReplaceAll(ev.Data, "\r", "\n")
	for _, line := range strings.Split(data, "\n") {
		_, _ = io.WriteString(w, "data: "+line+"\n")
	}
	_, _ = io.WriteString(w, "\n")
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func parseCommandLine(command string) ([]string, error) {
	var args []string
	var current strings.Builder
	var quote rune
	escaped := false
	for _, ch := range command {
		switch {
		case escaped:
			current.WriteRune(ch)
			escaped = false
		case ch == '\\':
			escaped = true
		case quote != 0:
			if ch == quote {
				quote = 0
			} else {
				current.WriteRune(ch)
			}
		case ch == '\'' || ch == '"':
			quote = ch
		case ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r':
			if current.Len() > 0 {
				args = append(args, current.String())
				current.Reset()
			}
		default:
			current.WriteRune(ch)
		}
	}
	if escaped {
		current.WriteRune('\\')
	}
	if quote != 0 {
		return nil, errors.New("unterminated quote in command")
	}
	if current.Len() > 0 {
		args = append(args, current.String())
	}
	if len(args) == 0 {
		return nil, errors.New("empty command")
	}
	return args, nil
}

func pathJoinSlash(base, rel string) string {
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(rel, "/")
}

func safeTruncate(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	// Ensure we don't split a multi-byte UTF-8 character
	end := maxBytes
	for end > 0 && end < len(s) && s[end]&0xC0 == 0x80 {
		end--
	}
	return s[:end]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
