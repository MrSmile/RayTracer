
SOURCE = main.cpp
HEADER = ray-tracer.h cl-helper.h
CLSOURCE = ray-tracer.h ray-tracer.cl shader.cl
FLAGS = -fno-exceptions -Wall -Wno-parentheses -Wno-long-long
LIBS = -lSDL -lGL -lOpenCL
PROGRAM = ray-tracer


debug: $(SOURCE) $(HEADER) shader
	g++ -g -O0 -DDEBUG $(FLAGS) $(SOURCE) $(LIBS) -o $(PROGRAM)

release: $(SOURCE) $(HEADER) shader
	g++ -O3 -flto -mtune=native -DNDEBUG $(FLAGS) $(SOURCE) $(LIBS) -o $(PROGRAM)

shader: $(CLSOURCE)
	rm -rf ~/.nv/ComputeCache

clean:
	rm $(PROGRAM)
