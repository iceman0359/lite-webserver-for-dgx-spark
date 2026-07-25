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
)

//go:embed static/index.html
var staticFiles embed.FS

const (
	maxUserSessions = 4
	maxSessionBytes = 256 * 1024
	maxSessionEvents = 256
	maxSessionSubscribers = 16
	maxFileEntries = 4000
	requestBodyLimit = 1 << 20
	maxTerminalInputBytes = 4096
	maxDownloadQueue = 64
)

type Config struct {
	BindHost          string   `json:"bindHost"`
	Port              int      `json:"port"`
	ContainerName     string   `json:"containerName"`
	AppRoot           string   `json:"appRoot"`
	DataRoot          string   `json:"dataRoot"`
	ContainerDataRoot string   `json:"containerDataRoot"`
	ModelsPath        string   `json:"modelsPath"`
	DownloadDirs      []string `json:"downloadDirs"`

	ClashControllerURL string `json:"clashControllerUrl"`
	ClashSecret        string `json:"clashSecret"`
	ClashStartCommand  string `json:"clashStartCommand"`
	ClashStopCommand   string `json:"clashStopCommand"`
	HTTPProxy          string `json:"httpProxy"`
	SocksProxy         string `json:"socksProxy"`
}

func defaultConfig() Config {
	return Config{
		BindHost:           "127.0.0.1",
		Port:               8848,
		ContainerName:      "comfyui",
		AppRoot:            "/home/simonsyoyo/Comfyui",
		DataRoot:           "/home/simonsyoyo/Comfyui/comfy_container_data",
		ContainerDataRoot:  "/opt/ComfyUI",
		ModelsPath:         "/home/simonsyoyo/Comfyui/ComfyUI/models",
		DownloadDirs:       []string{
			"/home/simonsyoyo/Comfyui/ComfyUI/models/checkpoints",
			"/home/simonsyoyo/Comfyui/ComfyUI/models/loras",
			"/home/simonsyoyo/Comfyui/ComfyUI/models/vae",
			"/home/simonsyoyo/Comfyui/comfy_container_data/custom_nodes",
			"/home/simonsyoyo/Comfyui/comfy_container_data/input",
			"/home/simonsyoyo/Comfyui/comfy_container_data/output",
			"/home/simonsyoyo/Comfyui/comfy_container_data/user",
		},
		ClashControllerURL: "http://127.0.0.1:9090",
		ClashStartCommand:  "clash-verge",
		ClashStopCommand:   "pkill -TERM -f 'clash-verge|Clash Verge'",
		HTTPProxy:          "http://127.0.0.1:7890",
		SocksProxy:         "socks5://127.0.0.1:7891",
	}
}

type Server struct {
	cfgMu sync.RWMutex
	cfg Config

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
	done chan error

	seq uint64
	byteCount int
	events []Event
	subs map[chan Event]struct{}
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

func main() {
	cfg := defaultConfig()
	if err := loadConfig("config.json", &cfg); err != nil {
		log.Printf("using default config: %v", err)
	}

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
		Addr: addr,
		Handler: localOnly(withSecurityHeaders(mux)),
		ReadHeaderTimeout: 5 * time.Second,
	}

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
	return os.WriteFile(path, data, 0600)
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
		data = data[:8192] + "\n[output truncated]"
	}
	ev := Event{Data: data, At: time.Now()}

	s.mu.Lock()
	s.seq++
	ev.Seq = s.seq
	s.events = append(s.events, ev)
	s.byteCount += len(ev.Data)
	for len(s.events) > 0 && (s.byteCount > maxSessionBytes || len(s.events) > maxSessionEvents) {
		s.byteCount -= len(s.events[0].Data)
		s.events = s.events[1:]
	}
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
	downloader.Append("[downloader] fixed output session.")
	downloader.Append("[downloader] 下载、更新和代理切换日志会输出在这里。")
	srv.sessions.Add(downloader)

	logs := NewSession("logs", "容器日志", srv.cfg.AppRoot, "fixed", true, false)
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
	mux.HandleFunc("/api/container/logs/start", srv.method("POST", srv.handleContainerLogsStart))

	mux.HandleFunc("/api/files/list", srv.method("GET", srv.handleFilesList))
	mux.HandleFunc("/api/files/delete", srv.method("POST", srv.handleFilesDelete))

	mux.HandleFunc("/api/terminal/sessions", srv.method("GET", srv.handleTerminalSessions))
	mux.HandleFunc("/api/terminal/create", srv.method("POST", srv.handleTerminalCreate))
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
		writeJSON(w, http.StatusOK, srv.currentConfig())
	case http.MethodPost:
		cfg := srv.currentConfig()
		if err := readJSON(r, &cfg); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
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
	srv.appendLogs("[docker] docker start "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "running", "output": out})
}

func (srv *Server) handleContainerStop(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	out, err := runCommand(r.Context(), "docker", "stop", name)
	srv.appendLogs("[docker] docker stop "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "stopped", "output": out})
}

func (srv *Server) handleContainerRestart(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	out, err := runCommand(r.Context(), "docker", "restart", name)
	srv.appendLogs("[docker] docker restart "+name+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "running", "output": out})
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

func (srv *Server) appendLogs(data string) {
	if logs, ok := srv.sessions.Get("logs"); ok {
		logs.Append(data)
	}
}

func (srv *Server) handleFilesList(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")
	if path == "" {
		path = srv.currentConfig().DataRoot
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
	srv.appendLogs("[docker] docker exec "+name+" rm -rf -- "+containerPath+"\n"+out)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"deleted": req.Path, "containerPath": containerPath})
}

func (srv *Server) hostToContainerPath(path string) (string, error) {
	cfg := srv.currentConfig()
	clean, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	root, err := filepath.Abs(cfg.DataRoot)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(root, clean)
	if err != nil {
		return "", err
	}
	if rel == "." || strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
		return "", fmt.Errorf("path must be inside data root: %s", cfg.DataRoot)
	}
	if strings.HasPrefix(filepath.Base(clean), ".") {
		return "", errors.New("hidden files are not managed")
	}
	return pathJoinSlash(cfg.ContainerDataRoot, filepath.ToSlash(rel)), nil
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
	session, err := srv.createShellSession(req.CWD, "命令行", []string{"/bin/bash"})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": session.ID})
}

func (srv *Server) handleTerminalDockerBash(w http.ResponseWriter, r *http.Request) {
	name := srv.currentConfig().ContainerName
	session, err := srv.createShellSession(srv.currentConfig().AppRoot, "Docker bash", []string{"docker", "exec", "-i", name, "bash"})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	session.mu.Lock()
	session.CWD = "/opt/ComfyUI"
	session.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"id": session.ID})
}

func (srv *Server) createShellSession(cwd, title string, argv []string) (*Session, error) {
	if srv.sessions.UserCount() >= maxUserSessions {
		return nil, fmt.Errorf("最多只能同时打开 %d 个普通命令行", maxUserSessions)
	}
	if len(argv) == 0 {
		return nil, errors.New("empty command")
	}
	checkedCWD, err := safeWorkingDir(cwd, srv.currentConfig().DataRoot)
	if err != nil {
		return nil, err
	}
	id := srv.sessions.NextUserID("term")
	session := NewSession(id, title+" "+id, checkedCWD, "user", false, true)
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = checkedCWD
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	session.cmd = cmd
	session.stdin = stdin
	session.done = make(chan error, 1)
	session.setRunning(true)
	session.Append("$ " + strings.Join(argv, " "))
	srv.sessions.Add(session)
	go pipeToSession(session, stdout)
	go pipeToSession(session, stderr)
	go func() {
		err := cmd.Wait()
		session.setRunning(false)
		session.Append(fmt.Sprintf("[process exited] %v", err))
		session.done <- err
	}()
	return session, nil
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
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
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
	if cmd == nil || cmd.Process == nil {
		session.Append("[control] " + control)
		return nil
	}
	pid := cmd.Process.Pid
	switch {
	case strings.HasPrefix(control, "Ctrl+C"):
		return syscall.Kill(-pid, syscall.SIGINT)
	case strings.HasPrefix(control, "Ctrl+Z"):
		return syscall.Kill(-pid, syscall.SIGTSTP)
	case strings.HasPrefix(control, "Ctrl+D"):
		if stdin != nil {
			return stdin.Close()
		}
	case strings.HasPrefix(control, "Ctrl+L"):
		session.Append("\x0c")
		return nil
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
	session.Append("[control] " + control)
	return nil
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
		session.CWD = task.Dir
		session.Append(fmt.Sprintf("[download #%d] %s", task.ID, task.Method))
		if task.Method == "wget" || task.Method == "git clone" {
			if err := os.MkdirAll(task.Dir, 0755); err != nil {
				session.Append("[error] mkdir: " + err.Error())
				continue
			}
		} else if _, err := safeWorkingDir(task.Dir, task.Dir); err != nil {
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
	srv.clashMu.RLock()
	state := map[string]any{
		"mode": srv.clashMode,
		"apiDetected": srv.clashAPI,
		"apiChecked": srv.clashAPIChecked,
		"appRunning": srv.clashRunning,
		"appChecked": srv.clashRunningChecked,
	}
	srv.clashMu.RUnlock()
	state["downloadProxy"] = srv.isDownloadProxyEnabled()
	writeJSON(w, http.StatusOK, state)
}

func (srv *Server) handleClashDetect(w http.ResponseWriter, r *http.Request) {
	cfg := srv.currentConfig()
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, strings.TrimRight(cfg.ClashControllerURL, "/")+"/version", nil)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if cfg.ClashSecret != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.ClashSecret)
	}
	client := http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	ok := err == nil && resp.StatusCode >= 200 && resp.StatusCode < 300
	if resp != nil {
		_ = resp.Body.Close()
	}
	srv.clashMu.Lock()
	srv.clashAPI = ok
	srv.clashAPIChecked = true
	if ok {
		srv.clashRunning = true
		srv.clashRunningChecked = true
	}
	srv.clashMu.Unlock()
	if downloads, ok := srv.sessions.Get("downloads"); ok {
		if err != nil {
			downloads.Append("[clash] detect failed: " + err.Error())
		} else {
			downloads.Append(fmt.Sprintf("[clash] detect api: %v", ok))
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": ok, "error": errString(err)})
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
	body, _ := json.Marshal(map[string]string{"mode": apiMode})
	cfg := srv.currentConfig()
	url := strings.TrimRight(cfg.ClashControllerURL, "/") + "/configs"
	httpReq, err := http.NewRequestWithContext(r.Context(), http.MethodPatch, url, bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if cfg.ClashSecret != "" {
		httpReq.Header.Set("Authorization", "Bearer "+cfg.ClashSecret)
	}
	client := http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(httpReq)
	apiOK := err == nil && resp.StatusCode >= 200 && resp.StatusCode < 300
	if resp != nil {
		_ = resp.Body.Close()
	}
	srv.clashMu.Lock()
	srv.clashMode = mode
	srv.clashAPI = apiOK
	srv.clashAPIChecked = true
	srv.clashMu.Unlock()
	if downloads, ok := srv.sessions.Get("downloads"); ok {
		downloads.Append(fmt.Sprintf("[clash] PATCH /configs mode=%s apiOK=%v error=%s", apiMode, apiOK, errString(err)))
	}
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"mode": mode, "apiOK": apiOK})
}

func (srv *Server) handleClashProxy(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Enabled bool `json:"enabled"`
	}
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	srv.dlMu.Lock()
	srv.downloadProxy = req.Enabled
	srv.dlMu.Unlock()
	if downloads, ok := srv.sessions.Get("downloads"); ok {
		downloads.Append(fmt.Sprintf("[proxy] download proxy enabled=%v", req.Enabled))
	}
	writeJSON(w, http.StatusOK, map[string]bool{"enabled": req.Enabled})
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
				session.Append("[read error] " + err.Error())
			}
			return
		}
	}
}

func runCommand(ctx context.Context, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
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
	for _, line := range strings.Split(ev.Data, "\n") {
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
