
SOURCE = main.cpp
HEADER =
FLAGS = -fno-exceptions -Wall -Wno-parentheses -Wno-long-long
LIBS = -lOpenCL
PROGRAM = ray-tracer


debug: $(SOURCE) $(HEADER)
	g++ -g -O0 -DDEBUG $(FLAGS) $(SOURCE) $(LIBS) -o $(PROGRAM)

release: $(SOURCE) $(HEADER)
	g++ -O3 -flto -mtune=native $(FLAGS) $(SOURCE) $(LIBS) -o $(PROGRAM)

clean:
	rm $(PROGRAM)
