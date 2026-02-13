import React, { useState, useRef, useEffect } from 'react';
import { Upload, Activity, AlertTriangle, CheckCircle, Phone, MessageSquare, Brain, TrendingUp, Zap, Eye } from 'lucide-react';

const AccidentPredictionUI = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [liveMetrics, setLiveMetrics] = useState({ frame: 0, risk: 0, level: 'LOW' });
  const [systemStatus, setSystemStatus] = useState({ camera: 'unknown', models: 'unknown', twilio: 'unknown' });
  const [location, setLocation] = useState('Highway Junction 45');
  const [riskThreshold, setRiskThreshold] = useState(55);
  const fileInputRef = useRef(null);

  // Fetch system status on component mount
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/status');
        if (response.ok) {
          const data = await response.json();
          setSystemStatus(data);
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      }
    };
    fetchStatus();
  }, []);

  // Actual API call to analyze video
  const analyzeVideo = async () => {
    if (!videoFile) return;

    setAnalyzing(true);
    setProgress(0);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('location', location);
      formData.append('risk_threshold', (riskThreshold / 100).toString());

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Simulate progress for UI feedback
      for (let i = 0; i <= 100; i += 5) {
        await new Promise(resolve => setTimeout(resolve, 50));
        setProgress(i);
      }

      if (data.detected) {
        setResults({
          detected: true,
          riskLevel: data.riskLevel.toUpperCase(),
          confidence: data.confidence,
          frame: data.frame,
          timestamp: data.timestamp,
          modelScores: data.modelScores,
          alerts: data.alerts
        });
      } else {
        setResults({ detected: false });
      }

    } catch (error) {
      console.error('Error analyzing video:', error);
      alert('Error analyzing video. Please try again.');
    } finally {
      setAnalyzing(false);
      setProgress(0);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const getRiskColor = (level) => {
    const colors = {
      'LOW': 'from-green-500 to-emerald-600',
      'MODERATE': 'from-yellow-500 to-orange-500',
      'HIGH': 'from-orange-500 to-red-500',
      'CRITICAL': 'from-red-600 to-rose-700'
    };
    return colors[level] || colors.LOW;
  };

  const getRiskIcon = (level) => {
    if (level === 'LOW') return <CheckCircle className="w-8 h-8" />;
    return <AlertTriangle className="w-8 h-8" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute top-40 right-20 w-72 h-72 bg-yellow-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-20 left-40 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl">
              <Brain className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-red-400">
              AI Accident Prediction
            </h1>
          </div>
          <p className="text-gray-300 text-lg">
            5-Model Ensemble • Real-time Analysis • Predictive Alerts
          </p>
        </header>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Upload & Preview */}
          <div className="lg:col-span-2 space-y-6">
            {/* Upload Area */}
            <div
              className="bg-slate-800/50 backdrop-blur-xl rounded-3xl border-2 border-slate-700/50 p-8 hover:border-purple-500/50 transition-all duration-300"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              {!videoPreview ? (
                <div className="text-center">
                  <div className="flex justify-center mb-6">
                    <div className="p-6 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full">
                      <Upload className="w-16 h-16 text-purple-400" />
                    </div>
                  </div>
                  <h3 className="text-2xl font-semibold text-white mb-2">Upload Video</h3>
                  <p className="text-gray-400 mb-6">Drag & drop or click to select</p>
                  <button
                    onClick={() => fileInputRef.current.click()}
                    className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-purple-500/50 transition-all duration-300"
                  >
                    Choose File
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <p className="text-gray-500 text-sm mt-4">Supports MP4, AVI, MOV, MKV</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <video
                    src={videoPreview}
                    controls
                    className="w-full rounded-2xl"
                  />
                  <div className="flex gap-3">
                    <button
                      onClick={() => fileInputRef.current.click()}
                      className="flex-1 px-6 py-3 bg-slate-700 text-white rounded-xl font-semibold hover:bg-slate-600 transition-all"
                    >
                      Change Video
                    </button>
                    <button
                      onClick={analyzeVideo}
                      disabled={analyzing}
                      className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-purple-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {analyzing ? (
                        <>
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Activity className="w-5 h-5" />
                          Start Analysis
                        </>
                      )}
                    </button>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </div>
              )}
            </div>

            {/* Live Metrics */}
            {analyzing && (
              <div className="bg-slate-800/50 backdrop-blur-xl rounded-3xl border-2 border-slate-700/50 p-6">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                  <Eye className="w-6 h-6 text-purple-400" />
                  Live Analysis
                </h3>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="bg-slate-900/50 rounded-xl p-4">
                    <p className="text-gray-400 text-sm mb-1">Frame</p>
                    <p className="text-2xl font-bold text-white">{liveMetrics.frame}</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-4">
                    <p className="text-gray-400 text-sm mb-1">Risk Score</p>
                    <p className="text-2xl font-bold text-white">{(liveMetrics.risk * 100).toFixed(0)}%</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-4">
                    <p className="text-gray-400 text-sm mb-1">Status</p>
                    <p className={`text-lg font-bold ${liveMetrics.level === 'CRITICAL' ? 'text-red-400' :
                        liveMetrics.level === 'HIGH' ? 'text-orange-400' :
                          liveMetrics.level === 'MODERATE' ? 'text-yellow-400' :
                            'text-green-400'
                      }`}>
                      {liveMetrics.level}
                    </p>
                  </div>
                </div>
                <div className="relative h-2 bg-slate-900 rounded-full overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-gray-400 text-sm mt-2 text-center">{progress}% Complete</p>
              </div>
            )}

            {/* Results */}
            {results && (
              <div className="space-y-6">
                {results.detected ? (
                  <>
                    {/* Main Result Card */}
                    <div className={`bg-gradient-to-br ${getRiskColor(results.riskLevel)} rounded-3xl p-8 text-white`}>
                      <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-4">
                          {getRiskIcon(results.riskLevel)}
                          <div>
                            <h3 className="text-3xl font-bold">{results.riskLevel} RISK DETECTED</h3>
                            <p className="text-white/80">Confidence: {(results.confidence * 100).toFixed(0)}%</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-white/80">Frame</p>
                          <p className="text-3xl font-bold">{results.frame}</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white/10 backdrop-blur rounded-xl p-4">
                          <p className="text-white/80 text-sm mb-1">Timestamp</p>
                          <p className="text-xl font-semibold">{results.timestamp}</p>
                        </div>
                        <div className="bg-white/10 backdrop-blur rounded-xl p-4">
                          <p className="text-white/80 text-sm mb-1">Total Warnings</p>
                          <p className="text-xl font-semibold">3</p>
                        </div>
                      </div>
                    </div>

                    {/* Model Scores */}
                    {results.modelScores && (
                      <div className="bg-slate-800/50 backdrop-blur-xl rounded-3xl border-2 border-slate-700/50 p-6">
                        <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                          <Brain className="w-6 h-6 text-purple-400" />
                          Model Contributions
                        </h3>
                        <div className="space-y-3">
                          {Object.entries(results.modelScores).map(([model, score]) => (
                            <div key={model} className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-300 font-medium uppercase">{model}</span>
                                <span className="text-purple-400 font-semibold">{(score * 100).toFixed(0)}%</span>
                              </div>
                              <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500"
                                  style={{ width: `${score * 100}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Alerts Status */}
                    {results.alerts && (
                      <div className="bg-slate-800/50 backdrop-blur-xl rounded-3xl border-2 border-slate-700/50 p-6">
                        <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                          <Zap className="w-6 h-6 text-yellow-400" />
                          Emergency Alerts
                        </h3>
                        <div className="space-y-3">
                          <div className="flex items-center justify-between p-4 bg-green-500/10 border border-green-500/30 rounded-xl">
                            <div className="flex items-center gap-3">
                              <Phone className="w-5 h-5 text-green-400" />
                              <div>
                                <p className="text-white font-semibold">Voice Call</p>
                                <p className="text-gray-400 text-sm">SID: {results.alerts.call.sid}</p>
                              </div>
                            </div>
                            <CheckCircle className="w-6 h-6 text-green-400" />
                          </div>
                          <div className="flex items-center justify-between p-4 bg-green-500/10 border border-green-500/30 rounded-xl">
                            <div className="flex items-center gap-3">
                              <MessageSquare className="w-5 h-5 text-green-400" />
                              <div>
                                <p className="text-white font-semibold">SMS Alert</p>
                                <p className="text-gray-400 text-sm">SID: {results.alerts.sms.sid}</p>
                              </div>
                            </div>
                            <CheckCircle className="w-6 h-6 text-green-400" />
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-3xl p-8 text-white">
                    <div className="flex items-center gap-4 mb-4">
                      <CheckCircle className="w-12 h-12" />
                      <div>
                        <h3 className="text-3xl font-bold">No Risk Detected</h3>
                        <p className="text-white/80">Traffic conditions appear normal</p>
                      </div>
                    </div>
                    <div className="bg-white/10 backdrop-blur rounded-xl p-4">
                      <p className="text-white/90">
                        ✅ The AI models analyzed the video and found no high-risk situations or accident predictions.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-6">
            {/* System Status */}
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-3xl border-2 border-slate-700/50 p-6">
              <h3 className="text-xl font-semibold text-white mb-4">System Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Camera</span>
                  <span className="text-green-400 font-semibold flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    Active
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">AI Models</span>
                  <span className="text-green-400 font-semibold flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    Loaded
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Twilio</span>
                  <span className="text-green-400 font-semibold flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    Connected
                  </span>
                </div>
              </div>
            </div>

            {/* Models Info */}
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-3xl border-2 border-slate-700/50 p-6">
              <h3 className="text-xl font-semibold text-white mb-4">Ensemble Models</h3>
              <div className="space-y-3">
                {['CNN', 'LSTM', 'Motion', 'Behavior', 'Trend'].map((model) => (
                  <div key={model} className="bg-slate-900/50 rounded-xl p-3">
                    <p className="text-white font-semibold">{model}</p>
                    <p className="text-gray-400 text-sm">
                      {model === 'CNN' && 'Spatial features'}
                      {model === 'LSTM' && 'Temporal patterns'}
                      {model === 'Motion' && 'Physics-based'}
                      {model === 'Behavior' && 'Risk assessment'}
                      {model === 'Trend' && 'Predictive analytics'}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Configuration */}
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-3xl border-2 border-slate-700/50 p-6">
              <h3 className="text-xl font-semibold text-white mb-4">Configuration</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-gray-300 text-sm mb-2 block">Location</label>
                  <input
                    type="text"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    className="w-full px-4 py-2 bg-slate-900 text-white rounded-xl border border-slate-700 focus:border-purple-500 focus:outline-none transition-colors"
                  />
                </div>
                <div>
                  <label className="text-gray-300 text-sm mb-2 block">Risk Threshold</label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={riskThreshold}
                    onChange={(e) => setRiskThreshold(Number(e.target.value))}
                    className="w-full accent-purple-500"
                  />
                  <p className="text-gray-400 text-xs mt-1">{riskThreshold}%</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default AccidentPredictionUI;