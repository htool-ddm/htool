#ifndef HTOOL_LOGGER
#define HTOOL_LOGGER
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

namespace htool {

// Meyer's singleton
// https://stackoverflow.com/a/1008289

enum class LogLevel : unsigned int {
    CRITICAL = 0,
    ERROR    = 10,
    WARNING  = 20,
    DEBUG    = 30,
    INFO     = 40,
};

std::string logging_level_to_string(LogLevel logging_level) {
    switch (logging_level) {
    case LogLevel::CRITICAL:
        return "[Htool critical] ";
        break;
    case LogLevel::ERROR:
        return "[Htool error]    ";
        break;
    case LogLevel::WARNING:
        return "[Htool warning]  ";
        break;
    case LogLevel::DEBUG:
        return "[Htool debug]    ";
        break;
    case LogLevel::INFO:
        return "[Htool info]     ";
        break;
    default:
        break;
    }
    return "";
}

class IObjectWriter {
  public:
    virtual void set_log_level(LogLevel)              = 0;
    virtual void write(LogLevel, const std::string &) = 0;
    virtual ~IObjectWriter() {}
};

class StandartOutputWriter : public IObjectWriter {
    LogLevel m_current_log_level = LogLevel::ERROR;

  public:
    void write(LogLevel log_level, const std::string &message) override {
        std::string prefix;
        if (log_level <= m_current_log_level) {
            std::cout << logging_level_to_string(log_level) + message << "\n";
        }
    }
    void set_log_level(LogLevel log_level) override { m_current_log_level = log_level; }
};

class Logger {
  public:
    void set_current_log_level(unsigned int log_level) { m_writer->set_log_level(static_cast<LogLevel>(log_level)); }
    void set_current_log_level(LogLevel log_level) { m_writer->set_log_level(log_level); }
    static Logger &get_instance() {
        static Logger instance;
        return instance;
    }

    void set_current_writer(std::shared_ptr<IObjectWriter> writer) { m_writer = writer; }

    void log(LogLevel log_level, std::string message) {
        m_writer->write(log_level, message);
    }

  protected:
    Logger()                          = default;
    Logger(Logger const &)            = delete;
    Logger &operator=(Logger &&)      = delete;
    Logger &operator=(const Logger &) = delete;
    virtual ~Logger() {}

  private:
    std::shared_ptr<IObjectWriter> m_writer = std::make_shared<StandartOutputWriter>();
};

} // namespace htool
#endif
