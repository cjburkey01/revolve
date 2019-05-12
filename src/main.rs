extern crate glium;

use glium::*;
use glium::glutin::*;
use glium::glutin::dpi::*;

fn main() {
    println!("Hello world!");

    let mut events_loop = EventsLoop::new();

    let display = create_display(640, 400, "Hello world!", &events_loop);

    loop {
        let mut close_requested = false;
        events_loop.poll_events(|e| {
            match e {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::CloseRequested => (close_requested = true),
                        _ => {},
                    }
                },
                _ => {},
            }
        });
        if close_requested {
            break;
        }

        update();

        let mut frame = display.draw();
        frame.clear_all((0.0, 0.0, 0.0, 1.0), 0.0, 0);
        draw(&display, &mut frame);
        frame.finish().unwrap();
    }
}

fn update() {}

fn draw(display: &Display, frame: &mut Frame) {}

fn create_display(width: i32, height: i32, title: &str, events_loop: &EventsLoop) -> Display {
    let window_builder = WindowBuilder::new()
        .with_dimensions(LogicalSize::new(width as f64, height as f64))
        .with_title(title);
    Display::new(window_builder, ContextBuilder::new(), events_loop).unwrap()
}
