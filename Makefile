.PHONY: clean libpedsim demo

all: libpedsim demo

libpedsim:
	make -C libpedsim

demo:
	make -C demo

clean:
	make -C libpedsim clean
	make -C demo clean
	-rm submission.tar.gz

submission: clean
	mkdir submit
	cp -r demo submit/
	cp -r libpedsim submit/
	cp Makefile submit/
	cp scenario.xml submit/
	cp scenario_box.xml submit/
	cp hugeScenario.xml submit/
	cp lab3-scenario.xml submit/
	tar -czvf submission.tar.gz submit
	rm -rf submit
