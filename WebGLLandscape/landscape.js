window.Seq = window.Seq || { }

Seq = function(obj, params, props, updateFunc, setPointFunc, initEase, otherEase) {
        
    s = this;

    s.rateMin =  params.rMin || 0;
    s.rateMax =  params.rMax || 60;
    s.delayRange =  params.delayRange || 0;
    s.delay =  params.delayRange || 0;
    s.start = 0;
    s.end = 0;
    s.duration = 0;

    s.props = [];
    props.forEach(function(prop, index, array) {
        s.props.push( {
            name: prop.name,
            min: prop.min || 0,
            max: prop.max || 1,
            setpoint: 0,
            range: prop.max - prop.min,
            prevValue: 0,
            value: 0,
            valueDiff: 0
        } );
    });


    s.obj = obj;
    s.timeout = null;
    s.ease = initEase || s.easeOutQuart;
    s.otherEase = otherEase || s.easeInOutSine;
    s.setPointFunc = setPointFunc || null;
    s.updateFunc = updateFunc || null;

    s.init = 10;

    s.next(true);

    if (Seq.sequences == null) {
        Seq.sequences = [];
        s.update();
    }
    Seq.sequences.push( s );
}

Seq.prototype.sequences = [];
Seq.prototype = {
   next: function (firstRun = false) {
        var s = this;
        var now = Date.now();
        if (!firstRun) {
            s.delay = Math.random() * s.delayRange * 1000;
            s.ease = s.otherEase;
        }
        s.start = now;
        s.end = s.start + ((s.init==0 ? ((s.rateMin + (Math.random() * (s.rateMax - s.rateMin)))) : s.init) * 1000);
        s.duration = s.end - s.start;

        s.props.forEach( function(prop, index, arr) {
            prop.prevValue = prop.value || 0;
            var rnd = (Math.random() * prop.range) + prop.min;
            prop.setpoint = s.setPointFunc!=null ? s.setPointFunc(s, prop.name, rnd) : rnd;
            prop.valueDiff = prop.setpoint - prop.prevValue;
         });
        
        s.init = 0;
        // console.log("Next: ID="+s.id+" prev="+s.prevValue+" setpoint="+s.setpoint+" diff="+s.valueDiff+" strt="+s.startTime+" end="+s.endTime+" dur="+s.duration);
    },

    update: function () {
        var s = this;
        var now = Date.now();
        Seq.sequences.forEach( function(seq, index, arr) {
            var timeLeft = seq.end - now;
            var t = (now - seq.start) / seq.duration;
            var isComplete = timeLeft < 0;
            if (isComplete) {
                if (timeLeft+seq.delay < 0)
                    seq.next();
            } else {
                seq.props.forEach( function(prop, index, arr) {
                    prop.value = prop.prevValue + (seq.ease(t) * prop.valueDiff)
                
                    if (seq.obj!=null) {
                        if (seq.updateFunc!=null) {
                            seq.updateFunc(seq, prop.value);
                        } else
                            seq.obj[prop.name] = prop.value;
                    }
                });
            }
        });

        setTimeout( function() {
            s.update();
        }, 50 );
    },

    easeInOutSine: function(t) { return t > 0.5 ? 4*Math.pow((t-1),3)+1 : 4*Math.pow(t,3); },
    easeInSine: function(t) { return 1 - Math.cos((t * Math.PI) / 2); },
    easeOutSine: function(t) { return Math.sin((t * Math.PI) / 2); },
    easeInOutQuad: function(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; },
    easeInQuad: function(t) { return t * t; },
    easeOutQuad: function(t) { return 1 - Math.pow(1 - t, 4); },
    easeInOutQuart: function(t) { return t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2; },
    easeInQuart: function(t) { return t * t * t * t; },
    easeOutQuart: function(t) { return 1 - (1 - t) * (1 - t); },
    easeInOutExp: function(t) { return t === 0 ? 0 : t === 1 ? 1 : t < 0.5 ? Math.pow(2, 20 * t - 10) / 2 : (2 - Math.pow(2, -20 * t + 10)) / 2; },
    easeInExp: function(t) { return t === 0 ? 0 : Math.pow(2, 10 * t - 10); },
    easeOutExp: function(t) { return t === 1 ? 1 : 1 - Math.pow(2, -10 * t); }
}

window.Landscape = window.Landscape || { }

Landscape = function( canvas, width, gridSize, gridHeight ) {

    ls = this;
    ls.canvas = canvas;
    ls.gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");

    if (!ls.gl instanceof WebGLRenderingContext) {
        console.log("is instance");
        ls.canvas.style.visibility = "hidden";
        ls.active = false;
        return;
    }
  
    ls.active = true;

    ls.gridSize = gridSize || 32; 
    ls.gs2 = (ls.gridSize / 2);
    ls.grid = [];
    ls.off = 2.5;
    ls.cellSize = width / gridSize;
    ls.lineWidth = 0.025;
    ls.lineCheckRatio = 0.0;
    ls.noiseHeight = gridHeight || 0;
    ls.logctr = 100;

    ls.fov = 40; // @ window width = 1000 therefore fov = 40/1000 * window.width
    ls.near = 1;
    ls.far = 500;

    ls.zSpeed = 0.0001;
    ls.curve = 0;
    ls.hill = 0;

    ls.hsvH = 0;
    ls.rgb = {r:0, g:0, b:0};
    ls.lineColor = { r:0.0, g:0.0, b:0.0 };
    ls.color1 = { r:0.0, g:0.0, b:0.0 };
    ls.color2 = { r:0.0, g:0.0, b:0.0 };

    ls.projMatrix = ls.get_projection(ls.fov, aspectRatio, ls.near, ls.far);
    console.log("Mat: "+ls.projMatrix);

    ls.transformMatrix = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
    ls.viewMatrix = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
    ls.noise = [ 0, 0, 0];
    ls.noiseOffset = [ 0, 0 ];

    ls.rotateX(ls.viewMatrix, tilt);
    ls.translate(ls.viewMatrix, 0, 8, -90);

    ls.prevTime = 0;

    ls.resize(ls.canvas);

    // Generate grid vertices and indices
    ls.vertices = [];
    ls.uvs = [];
    ls.indices = [];

    var x, y, base;
    var tw = ls.gridSize + 1;
    var numIndices = 0;
    var maxSize = ls.gridSize * ls.cellSize;
    console.log("MaxSize="+maxSize+" numSteps="+ls.gridSize+" cellSize="+ls.cellSize);
    var index = 0;
    var uvInd = 0;
    for (var yi=0; yi<=ls.gridSize; yi++) {
        for (var xi=0; xi<=ls.gridSize; xi++) {
            x = (xi / ls.gridSize - .5) * maxSize;
            y = (yi / ls.gridSize - .5) * maxSize;
            
            ls.vertices[index++] = x;
            ls.vertices[index++] = (xi==0 & yi==0) ? 50 : 0;
            ls.vertices[index++] = y;

            ls.uvs[uvInd++] = (xi / ls.gridSize);
            ls.uvs[uvInd++] = (1 - yi / ls.gridSize);

            if (xi != ls.gridSize && yi != ls.gridSize) {
                base = xi + yi * tw;
                var colInd = numIndices;
                ls.indices[numIndices++] = base;
                ls.indices[numIndices++] = base + tw + 1;
                ls.indices[numIndices++] = base + tw;
                ls.indices[numIndices++] = base;
                ls.indices[numIndices++] = base + 1;
                ls.indices[numIndices++] = base + tw + 1;
            }
        }
    }

    // Create and store data into vertex buffer
    ls.vertex_buffer = ls.gl.createBuffer ();
    ls.gl.bindBuffer(ls.gl.ARRAY_BUFFER, ls.vertex_buffer);
    ls.gl.bufferData(ls.gl.ARRAY_BUFFER, new Float32Array(ls.vertices), ls.gl.STATIC_DRAW);

    // Create and store data into uv buffer
    ls.uv_buffer = ls.gl.createBuffer ();
    ls.gl.bindBuffer(ls.gl.ARRAY_BUFFER, ls.uv_buffer);
    ls.gl.bufferData(ls.gl.ARRAY_BUFFER, new Float32Array(ls.uvs), ls.gl.STATIC_DRAW);

    // Create and store data into index buffer
    ls.index_buffer = ls.gl.createBuffer ();
    ls.gl.bindBuffer(ls.gl.ELEMENT_ARRAY_BUFFER, ls.index_buffer);
    ls.gl.bufferData(ls.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(ls.indices), ls.gl.STATIC_DRAW);

    /*=================== Shaders =========================*/

    var vertCode = `
        attribute vec3 position;
        attribute vec2 uv;
        
        uniform mat4 uProjectionMatrix;
        uniform mat4 uViewMatrix;
        uniform mat4 uTransformMatrix;
        uniform mediump float uCellSize;
        uniform mediump float uNumSteps;
        uniform float uCurve;
        uniform float uHill;
        uniform vec3 uNoise;
        uniform float uNoiseHeight;
        uniform mediump vec2 uNoiseOffset;
        
        varying vec2 vUV;

        vec2 hash( vec2 p )
        {
            p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
            return -1.0 + 2.0*fract(sin(p)*43758.5453123);
        }

        float noise( in vec2 p )
        {
            const float K1 = 0.366025404; // (sqrt(3)-1)/2;
            const float K2 = 0.211324865; // (3-sqrt(3))/6;

            vec2  i = floor( p + (p.x+p.y)*K1 );
            vec2  a = p - i + (i.x+i.y)*K2;
            float m = step(a.y,a.x); 
            vec2  o = vec2(m,1.0-m);
            vec2  b = a - o + K2;
            vec2  c = a - 1.0 + 2.0*K2;
            vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
            vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
            return dot( n, vec3(70.0) );
        }

        void main(void) {
            // Create an offset for the noise - floor avoids a skipping effects and syncs with the grid
            float noiseOffsetY = floor( uNoiseOffset.y * uNumSteps) / uNumSteps;
            
            // Use the uv coords and a parameterised offset to get the scrolling noise coords
            vec2 noisePos = uv + vec2(0., noiseOffsetY);

            // Create a scale based on the uv.x so it flattens towards the center of the grid (horizontally)
            float scale = abs(sin((uv.x - 0.5) / 1.55)) + 0.02;
            
            // Generate 3 frequencies of noise
            float nA = noise(noisePos * vec2(32.));
            float nB = noise(noisePos * vec2(8.));
            float nC = noise(noisePos * vec2(2.));

            // Combine those 3 noises with parameterised weighting (each 0-1)
            float noise = (nA*uNoise.x)+(nB*uNoise.y)+(nC*uNoise.z);

            // Convert the noise into a height with an overall heigh scale and the UV.x scale
            float height = noise * uNoiseHeight * scale * scale * 4.; 

            float posOffset = mod(uNoiseOffset.y * uCellSize * uNumSteps, uCellSize);

            float x2 = posOffset / uNumSteps * uv.y * uCurve / uCellSize * 2.;
            float xOff = -(x2) + (uv.y * uv.y * uCurve);
            float y2 = posOffset / uNumSteps * uv.y * uHill / uCellSize * 2.;
            float yOff = -(y2) + (uv.y * uv.y * uHill);
            
            gl_Position = uProjectionMatrix * uViewMatrix * uTransformMatrix * vec4( position.x + xOff, height + yOff, position.z + posOffset, 1.);
            vUV = uv;
        }
   `;

    var fragCode = `
        precision mediump float;

        uniform float uNumSteps;
        uniform float uCellSize;
        uniform float uLineWidth;
        uniform float uLineCheckRatio;
        uniform vec3 uLineColor;
        uniform vec3 uColor1;
        uniform vec3 uColor2;
        uniform vec2 uNoiseOffset;
        
        varying vec2 vUV;

        void main(void) {
            float edge = 0.02;
            float edgeMin = 0.-(edge * 0.5);
            float edgeMax = 0.+(edge * 0.5);

            float noiseOffsetY = floor( uNoiseOffset.y * uNumSteps) / uNumSteps;
            float s = floor( vUV.y * uNumSteps) / uNumSteps;

            float lineX = (clamp(mod(vUV.x * uNumSteps + (uLineWidth * 0.5), 1.)-uLineWidth, edgeMin, edgeMax) / edge);
            float lineY = (clamp(mod((vUV.y + noiseOffsetY) * uNumSteps + (uLineWidth * 0.5), 1.)-uLineWidth, edgeMin, edgeMax) / edge);

            float l1 = smoothstep(0., 1., lineX) + smoothstep(0., 1., lineY);
            float l2 = smoothstep(0., 1., lineX + lineY); 
            float l = mix(l2, l1, uLineCheckRatio);

            float off = mod((noiseOffsetY * uCellSize) * uNumSteps, uCellSize) / (uCellSize * uNumSteps);
            vec3 col = mix(mix(uLineColor, uColor1, s+off), mix(uColor2, uColor1, s+off), l);

            gl_FragColor = vec4( col, 1.);
        }
    `;

    var vertShader = ls.gl.createShader(ls.gl.VERTEX_SHADER);
    ls.gl.shaderSource(vertShader, vertCode);
    ls.gl.compileShader(vertShader);

    var fragShader = ls.gl.createShader(ls.gl.FRAGMENT_SHADER);
    ls.gl.shaderSource(fragShader, fragCode);
    ls.gl.compileShader(fragShader);

    var shaderProgram = ls.gl.createProgram();
    ls.gl.attachShader(shaderProgram, vertShader);
    ls.gl.attachShader(shaderProgram, fragShader);
    ls.gl.linkProgram(shaderProgram);

    /* ====== Associating attributes to the shaders =====*/
    ls.uProjectionMatrix = ls.gl.getUniformLocation(shaderProgram, "uProjectionMatrix");
    ls.uViewMatrix = ls.gl.getUniformLocation(shaderProgram, "uViewMatrix");
    ls.uTransformMatrix = ls.gl.getUniformLocation(shaderProgram, "uTransformMatrix");
    ls.uCellSize = ls.gl.getUniformLocation(shaderProgram, "uCellSize");
    ls.uLineWidth = ls.gl.getUniformLocation(shaderProgram, "uLineWidth");
    ls.uLineCheckRatio = ls.gl.getUniformLocation(shaderProgram, "uLineCheckRatio");
    ls.uNoise = ls.gl.getUniformLocation(shaderProgram, "uNoise");
    ls.uNoiseHeight = ls.gl.getUniformLocation(shaderProgram, "uNoiseHeight");
    ls.uNoiseOffset = ls.gl.getUniformLocation(shaderProgram, "uNoiseOffset");

    ls.uNumSteps = ls.gl.getUniformLocation(shaderProgram, "uNumSteps");
    ls.uCurve = ls.gl.getUniformLocation(shaderProgram, "uCurve");
    ls.uHill = ls.gl.getUniformLocation(shaderProgram, "uHill");
    ls.uLineColor = ls.gl.getUniformLocation(shaderProgram, "uLineColor");
    ls.uColor1 = ls.gl.getUniformLocation(shaderProgram, "uColor1");
    ls.uColor2 = ls.gl.getUniformLocation(shaderProgram, "uColor2");

    // Position
    ls.gl.bindBuffer(ls.gl.ARRAY_BUFFER, ls.vertex_buffer);
    var position = ls.gl.getAttribLocation(shaderProgram, "position");
    ls.gl.vertexAttribPointer(position, 3, ls.gl.FLOAT, false, 0, 0);
    ls.gl.enableVertexAttribArray(position);
    
    // UVs
    ls.gl.bindBuffer(ls.gl.ARRAY_BUFFER, ls.uv_buffer);
    var uv = ls.gl.getAttribLocation(shaderProgram, "uv");
    ls.gl.vertexAttribPointer(uv, 2, ls.gl.FLOAT, false, 0, 0) ;
    ls.gl.enableVertexAttribArray(uv);

    ls.gl.useProgram(shaderProgram);

    ls.drawGL(0);
}

Landscape.prototype = {

    get_projection: function (angle, aR, zMin, zMax) {
        console.log("Proj: ang="+angle+" aR:"+aR+" near="+zMin+" far="+zMax);
        var f = 1.0 / Math.tan((angle * Math.PI / 180) / 2);
        var rangeInv = 1 / (zMin - zMax);
        return [
            f/aR, 0 , 0, 0,
            0, f, 0, 0,
            0, 0, (zMin + zMax) * rangeInv, -1,
            0, 0, zMin * zMax * rangeInv * 2, 0 
        ];
    },

    translate: function(m, x, y, z) {
        m[12] += x;
        m[13] += y;
        m[14] += z;
    },

    rotateZ: function(m, angle) {
        var c = Math.cos(angle);
        var s = Math.sin(angle);
        var mv0 = m[0], mv4 = m[4], mv8 = m[8];

        m[0] = c*m[0]-s*m[1];
        m[4] = c*m[4]-s*m[5];
        m[8] = c*m[8]-s*m[9];

        m[1]=c*m[1]+s*mv0;
        m[5]=c*m[5]+s*mv4;
        m[9]=c*m[9]+s*mv8;
    },

    rotateX: function(m, angle) {
        var c = Math.cos(angle);
        var s = Math.sin(angle);
        var mv1 = m[1], mv5 = m[5], mv9 = m[9];

        m[1] = m[1]*c-m[2]*s;
        m[5] = m[5]*c-m[6]*s;
        m[9] = m[9]*c-m[10]*s;

        m[2] = m[2]*c+mv1*s;
        m[6] = m[6]*c+mv5*s;
        m[10] = m[10]*c+mv9*s;
    },

    rotateY: function(m, angle) {
        var c = Math.cos(angle);
        var s = Math.sin(angle);
        var mv0 = m[0], mv4 = m[4], mv8 = m[8];

        m[0] = c*m[0]+s*m[2];
        m[4] = c*m[4]+s*m[6];
        m[8] = c*m[8]+s*m[10];

        m[2] = c*m[2]-s*mv0;
        m[6] = c*m[6]-s*mv4;
        m[10] = c*m[10]-s*mv8;
    },

    /*================= Drawing ===========================*/

    drawGL: function(now) {

        //fpsElem = document.getElementById("fps");
        now *= 0.001;
        var deltaTime = now - ls.prevTime;
        ls.prevTime = now;
        //const fps = 1 / deltaTime;
        //fpsElem.textContent = "FPS: "+fps.toFixed(1);

        ls.resize(ls.gl.canvas);

        ls.gl.enable(ls.gl.DEPTH_TEST);
        ls.gl.depthFunc(ls.gl.LEQUAL);
        ls.gl.clearColor(ls.color1.r, ls.color1.g, ls.color1.b, 1);
        ls.gl.clearDepth(1.0);

        ls.gl.viewport(0.0, 0.0, ls.canvas.width, ls.canvas.height);
        ls.gl.clear(ls.gl.COLOR_BUFFER_BIT | ls.gl.DEPTH_BUFFER_BIT);
        ls.gl.uniformMatrix4fv(ls.uProjectionMatrix, false, ls.projMatrix);
        ls.gl.uniformMatrix4fv(ls.uViewMatrix, false, ls.viewMatrix);
        ls.gl.uniformMatrix4fv(ls.uTransformMatrix, false, ls.transformMatrix);
        ls.gl.uniform1f(ls.uCellSize, ls.cellSize);
        ls.gl.uniform1f(ls.uLineWidth, ls.lineWidth);
        ls.gl.uniform1f(ls.uLineCheckRatio, ls.lineCheckRatio);
        ls.gl.uniform3fv(ls.uNoise, ls.noise);
        ls.gl.uniform1f(ls.uNoiseHeight, ls.noiseHeight);
        ls.gl.uniform2fv(ls.uNoiseOffset, ls.noiseOffset);
        ls.gl.uniform1f(ls.uNumSteps, ls.gridSize);
        ls.gl.uniform1f(ls.uCurve, ls.curve);
        ls.gl.uniform1f(ls.uHill, ls.hill);
        ls.gl.uniform3fv(ls.uLineColor, [ ls.lineColor.r, ls.lineColor.g, ls.lineColor.b ] );
        ls.gl.uniform3fv(ls.uColor1, [ ls.color1.r, ls.color1.g, ls.color1.b ]);
        ls.gl.uniform3fv(ls.uColor2, [ ls.color2.r, ls.color2.g, ls.color2.b ]);
        ls.gl.bindBuffer(ls.gl.ELEMENT_ARRAY_BUFFER, ls.index_buffer);
        ls.gl.drawElements(ls.gl.TRIANGLES, ls.indices.length, ls.gl.UNSIGNED_SHORT, 0);

        ls.noiseOffset[1] += ls.zSpeed;
        window.requestAnimationFrame(ls.drawGL);
    },

    resize: function(canvas) {
        // Lookup the size the browser is displaying the canvas in CSS pixels.
        const displayWidth  = window.innerWidth;
        const displayHeight = window.innerHeight;

        // Check if the canvas is not the same size.
        const needResize = canvas.width  !== displayWidth || canvas.height !== displayHeight;

        if (needResize) {
            // Make the canvas the same size
            canvas.width = displayWidth;
            canvas.height = displayHeight;
            console.log("Resize: "+canvas.width+"/"+canvas.height);
            ls.projMatrix = ls.get_projection(ls.fov, aspectRatio, ls.near, ls.far);
        }

        return needResize;
    }
}
